package instance

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/hetznercloud/hcloud-go/v2/hcloud"
)

// Drift checks run far more often than anything else that reads servers: Karpenter
// core re-runs drift detection for a NodeClaim on every event of any pod on its node
// (nodeclaim.disruption watches Pods). On a pool of ephemeral CI runners that is
// thousands of checks an hour, and one GET /servers/{id} each used the whole hcloud
// project budget on its own (#126). GetForDrift serves them from one shared list of
// this cluster's servers instead, so their cost no longer depends on pod churn.
const (
	// DefaultSnapshotTTL bounds how stale a drift check may be while the API answers.
	// Drift is not urgent: core already requeues every NodeClaim's drift check every
	// few minutes.
	DefaultSnapshotTTL = time.Minute
	// snapshotMaxAge bounds how stale a drift check may be while refreshes fail. Up to
	// this age the previous copy is served; beyond it every check returns the refresh
	// error, so IsDrifted makes no disruption decision on old state. It is longer than
	// snapshotBackoffMax, so one slow retry does not fail checks.
	snapshotMaxAge = 10 * time.Minute
	// After a failed request no request of the same kind is made for
	// snapshotBackoffMin, doubling per consecutive failure up to snapshotBackoffMax, so
	// a 429 does not turn every pod event into another request.
	snapshotBackoffMin = 5 * time.Second
	snapshotBackoffMax = 5 * time.Minute
)

// serverSnapshot is a cluster-scoped, TTL-bounded copy of the servers List returns.
// One refresh runs at a time and every concurrent caller waits for it. A failed refresh
// keeps the previous copy for at most snapshotMaxAge; it is never replaced by an empty
// one.
type serverSnapshot struct {
	list  func(ctx context.Context) ([]*hcloud.Server, error)
	get   func(ctx context.Context, id int64) (*hcloud.Server, error)
	ttl   time.Duration
	nowFn func() time.Time

	mu        sync.Mutex
	servers   map[int64]*hcloud.Server
	fetchedAt time.Time
	valid     bool
	failures  int
	retryAt   time.Time
	lastErr   error
	inflight  *snapshotRefresh
	// lookups holds the live lookup of each ID the list did not contain (a server
	// created after the refresh, or one without this cluster's labels). A successful
	// answer, nil meaning "not found", is reused until the next refresh; a failed one
	// is returned until its retry time, which survives refreshes. Concurrent callers
	// for one ID share a single in-flight request.
	lookups map[int64]*serverLookup
}

type snapshotRefresh struct {
	done chan struct{}
	err  error
}

// serverLookup is one live GET for an ID missing from the snapshot. Its result fields
// are written under serverSnapshot.mu before done is closed.
type serverLookup struct {
	done     chan struct{}
	server   *hcloud.Server
	err      error
	failures int
	retryAt  time.Time
}

func (l *serverLookup) finished() bool {
	select {
	case <-l.done:
		return true
	default:
		return false
	}
}

func newServerSnapshot(
	list func(ctx context.Context) ([]*hcloud.Server, error),
	get func(ctx context.Context, id int64) (*hcloud.Server, error),
	ttl time.Duration, nowFn func() time.Time,
) *serverSnapshot {
	return &serverSnapshot{list: list, get: get, ttl: ttl, nowFn: nowFn}
}

// Get returns the server with this ID, or nil when it does not exist.
func (s *serverSnapshot) Get(ctx context.Context, id int64) (*hcloud.Server, error) {
	if err := s.ensureFresh(ctx); err != nil {
		return nil, err
	}

	s.mu.Lock()
	if server, ok := s.servers[id]; ok {
		s.mu.Unlock()
		return server, nil
	}
	l := s.lookups[id]
	switch {
	case l == nil:
		l = s.startLookupLocked(ctx, id, 0)
	case !l.finished():
		// Join the request already in flight.
	case l.err == nil:
		s.mu.Unlock()
		return l.server, nil
	case s.nowFn().Before(l.retryAt):
		err, retryAt := l.err, l.retryAt
		s.mu.Unlock()
		return nil, fmt.Errorf("server %d lookup backing off until %s: %w", id, retryAt.Format(time.RFC3339), err)
	default:
		l = s.startLookupLocked(ctx, id, l.failures)
	}
	s.mu.Unlock()

	select {
	case <-l.done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return l.server, l.err
}

// startLookupLocked registers and starts a live lookup of id; s.mu must be held.
// failures carries the consecutive failure count of the lookup it replaces.
func (s *serverSnapshot) startLookupLocked(ctx context.Context, id int64, failures int) *serverLookup {
	l := &serverLookup{done: make(chan struct{}), failures: failures}
	if s.lookups == nil {
		s.lookups = map[int64]*serverLookup{}
	}
	s.lookups[id] = l
	// The lookup serves every waiter, so the first caller's cancellation must not end
	// it; it keeps the caller's values only.
	go s.lookup(context.WithoutCancel(ctx), id, l)
	return l
}

func (s *serverSnapshot) lookup(ctx context.Context, id int64, l *serverLookup) {
	ctx, cancel := context.WithTimeout(ctx, refreshTimeout(s.ttl))
	defer cancel()
	server, err := s.get(ctx, id)

	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil {
		l.failures++
		l.retryAt = s.nowFn().Add(backoffFor(l.failures))
		l.err = fmt.Errorf("looking up server %d: %w", id, err)
	} else {
		l.server = server
		l.failures = 0
	}
	close(l.done)
}

// ensureFresh returns once the snapshot is younger than the TTL, or once a stale copy
// younger than snapshotMaxAge may be served because a refresh failed or is not allowed
// yet (backoff). Otherwise it returns the last refresh error.
func (s *serverSnapshot) ensureFresh(ctx context.Context) error {
	s.mu.Lock()
	now := s.nowFn()
	if s.valid && now.Sub(s.fetchedAt) < s.ttl {
		s.mu.Unlock()
		return nil
	}
	if now.Before(s.retryAt) {
		err := s.staleErrLocked(now, fmt.Errorf("server snapshot unavailable until %s: %w", s.retryAt.Format(time.RFC3339), s.lastErr))
		s.mu.Unlock()
		return err
	}
	call := s.inflight
	if call == nil {
		call = &snapshotRefresh{done: make(chan struct{})}
		s.inflight = call
		s.mu.Unlock()
		// The refresh serves every waiter, so the first caller's cancellation must not
		// end it; it keeps the caller's values only.
		go s.refresh(context.WithoutCancel(ctx), call)
	} else {
		s.mu.Unlock()
	}

	select {
	case <-call.done:
	case <-ctx.Done():
		return ctx.Err()
	}
	if call.err == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.staleErrLocked(s.nowFn(), call.err)
}

// staleErrLocked returns nil when the previous copy is young enough to serve after a
// failed or deferred refresh, and err otherwise; s.mu must be held.
func (s *serverSnapshot) staleErrLocked(now time.Time, err error) error {
	if !s.valid {
		return err
	}
	if age := now.Sub(s.fetchedAt); age >= snapshotMaxAge {
		return fmt.Errorf("server snapshot is %s old, beyond the %s limit for drift checks: %w", age.Round(time.Second), snapshotMaxAge, err)
	}
	return nil
}

func (s *serverSnapshot) refresh(ctx context.Context, call *snapshotRefresh) {
	ctx, cancel := context.WithTimeout(ctx, refreshTimeout(s.ttl))
	defer cancel()
	servers, err := s.list(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.nowFn()
	if err != nil {
		s.failures++
		s.retryAt = now.Add(backoffFor(s.failures))
		s.lastErr = err
		call.err = fmt.Errorf("refreshing server snapshot: %w", err)
	} else {
		byID := make(map[int64]*hcloud.Server, len(servers))
		for _, server := range servers {
			if server != nil {
				byID[server.ID] = server
			}
		}
		s.servers = byID
		s.fetchedAt = now
		s.valid = true
		s.failures = 0
		s.retryAt = time.Time{}
		s.lastErr = nil
		// Answers belong to the previous snapshot. Failed lookups still in backoff
		// stay, so a refresh does not reset their request budget; lookups in flight
		// still answer their waiters but are not reused.
		var kept map[int64]*serverLookup
		for id, l := range s.lookups {
			if l.finished() && l.err != nil && now.Before(l.retryAt) {
				if kept == nil {
					kept = map[int64]*serverLookup{}
				}
				kept[id] = l
			}
		}
		s.lookups = kept
	}
	s.inflight = nil
	close(call.done)
}

// backoffFor returns the wait after the given number of consecutive failures.
func backoffFor(failures int) time.Duration {
	backoff := snapshotBackoffMin << min(failures-1, 16)
	if backoff > snapshotBackoffMax || backoff <= 0 {
		return snapshotBackoffMax
	}
	return backoff
}

// refreshTimeout bounds one request independently of the caller that started it.
func refreshTimeout(ttl time.Duration) time.Duration {
	if ttl < 30*time.Second {
		return 30 * time.Second
	}
	return ttl
}
