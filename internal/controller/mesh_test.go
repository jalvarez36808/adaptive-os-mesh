package controller

import (
	"sync"
	"testing"
	pb "github.com/groovy-byte/agent-mesh-core/proto"
)

func TestMeshRegistryMetrics(t *testing.T) {
	r := NewMeshRegistry()
	
	// 1. Register Agent
	agentID := "test-agent"
	r.RegisterAgent(&pb.HandshakeRequest{
		AgentId: agentID,
		InitialRole: pb.AgentRole_OPERATIONAL,
	})

	// 2. Record multiple metrics
	r.RecordMetrics(agentID, 100.0, 50, 10.0)
	r.RecordMetrics(agentID, 200.0, 150, 12.0)

	// 3. Verify Stats
	stats := r.GetStatsSummary()
	if len(stats) != 1 {
		t.Fatalf("Expected 1 agent in stats, got %d", len(stats))
	}

	s := stats[0]
	if s.ID != agentID {
		t.Errorf("Expected ID %s, got %s", agentID, s.ID)
	}
	if s.AvgLatency != 150.0 {
		t.Errorf("Expected avg latency 150.0, got %f", s.AvgLatency)
	}
	if s.Requests != 2 {
		t.Errorf("Expected 2 requests, got %d", s.Requests)
	}
	if s.Tokens != 200 {
		t.Errorf("Expected 200 tokens, got %d", s.Tokens)
	}
}

func TestMeshRegistryContribution(t *testing.T) {
	r := NewMeshRegistry()
	
	source := "scout"
	target := "coder"
	
	// 1. Record contribution
	r.RecordContribution(source, target, 0.5)
	r.RecordContribution(source, target, 0.25)
	
	// 2. Verify Detail
	detail := r.GetContributionDetail(source)
	score, ok := detail[target]
	if !ok {
		t.Fatalf("Expected contribution to %s not found", target)
	}
	if score != 0.75 {
		t.Errorf("Expected combined score 0.75, got %f", score)
	}
	
	// 3. Verify non-existent
	if len(r.GetContributionDetail("unknown")) != 0 {
		t.Errorf("Expected empty detail for unknown agent")
	}
}

func TestMeshRegistryContributionConcurrency(t *testing.T) {
	r := NewMeshRegistry()
	source := "parallel-source"
	target := "target"
	
	const count = 100
	const increment = 0.01
	
	var wg sync.WaitGroup
	wg.Add(count)
	
	for i := 0; i < count; i++ {
		go func() {
			defer wg.Done()
			r.RecordContribution(source, target, increment)
		}()
	}
	
	wg.Wait()
	
	detail := r.GetContributionDetail(source)
	// 100 * 0.01 should be 1.0 (float precision warning, but 0.01 is usually safe in small counts)
	if detail[target] < 0.99 || detail[target] > 1.01 {
		t.Errorf("Expected score ~1.0 after concurrent updates, got %f", detail[target])
	}
}

func TestMeshRegistryTaskAudit(t *testing.T) {
	r := NewMeshRegistry()
	agentID := "audit-agent"
	r.RegisterAgent(&pb.HandshakeRequest{AgentId: agentID})

	// 1. Record successes
	r.RecordTaskResult(agentID, true, "TASK_1", 3)
	r.RecordTaskResult(agentID, true, "TASK_2", 2)

	// 2. Record 6 failures (testing circular buffer of 5)
	failures := []string{"FAIL_1", "FAIL_2", "FAIL_3", "FAIL_4", "FAIL_5", "FAIL_6"}
	for _, f := range failures {
		r.RecordTaskResult(agentID, false, f, 1)
	}

	agent, _ := r.GetAgent(agentID)
	if agent.ToolCalls != 11 { // 3 + 2 + 6*1
		t.Errorf("Expected 11 total tool calls, got %d", agent.ToolCalls)
	}

	if len(agent.FailedTasks) != 5 {
		t.Errorf("Expected 5 failed tasks in log, got %d", len(agent.FailedTasks))
	}

	// Latest failure should be at the front
	if agent.FailedTasks[0] != "FAIL_6" {
		t.Errorf("Expected most recent failure 'FAIL_6' at front, got %s", agent.FailedTasks[0])
	}
	
	// Oldest failure (FAIL_1) should have been pushed out
	for _, f := range agent.FailedTasks {
		if f == "FAIL_1" {
			t.Errorf("Old failure 'FAIL_1' should have been evicted from log")
		}
	}
}
