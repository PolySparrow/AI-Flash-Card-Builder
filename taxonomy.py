# taxonomy.py  (enriched version)

TAXONOMY = {
    "1": {
        "domain": "Domain1: Agentic Architecture & Orchestration",
        "tasks": {
            "1.1": {
                "statement": (
                    "Design and implement agentic loops for autonomous task execution"
                ),
                "description": (
                    "Agentic loop lifecycle: sending requests to Claude, "
                    "inspecting stop_reason (tool_use vs end_turn), executing "
                    "tools, and returning results. Tool results appended to "
                    "conversation history between iterations."
                ),
                "keywords": [
                    "agentic loop",
                    "stop_reason",
                    "tool_use",
                    "end_turn",
                    "loop termination",
                    "iteration",
                    "autonomous task",
                ],
            },
            "1.2": {
                "statement": (
                    "Orchestrate multi-agent systems with coordinator-subagent patterns"
                ),
                "description": (
                    "Hub-and-spoke architecture where a coordinator manages "
                    "inter-subagent communication, error handling, and routing. "
                    "Subagents have isolated context. Coordinator handles task "
                    "decomposition, delegation, result aggregation."
                ),
                "keywords": [
                    "coordinator",
                    "subagent",
                    "hub-and-spoke",
                    "multi-agent",
                    "orchestration",
                    "task decomposition",
                    "delegation",
                    "aggregation",
                ],
            },
            "1.3": {
                "statement": (
                    "Configure subagent invocation, context passing, and spawning"
                ),
                "description": (
                    "Task tool as the mechanism for spawning subagents. "
                    "allowedTools must include Task. Subagent context must be "
                    "explicitly provided. AgentDefinition config. Fork-based "
                    "session management. Parallel subagents via multiple Task "
                    "tool calls in one response."
                ),
                "keywords": [
                    "Task tool",
                    "allowedTools",
                    "spawn",
                    "AgentDefinition",
                    "context passing",
                    "parallel subagents",
                    "fork_session",
                    "prompt injection",
                ],
            },
            "1.4": {
                "statement": (
                    "Implement multi-step workflows with enforcement and handoff patterns"
                ),
                "description": (
                    "Programmatic enforcement vs prompt-based guidance. "
                    "Prerequisite gates blocking downstream tool calls. "
                    "Structured handoff protocols for escalation including "
                    "customer details, root cause, recommended actions."
                ),
                "keywords": [
                    "prerequisite",
                    "programmatic enforcement",
                    "handoff",
                    "escalation",
                    "workflow ordering",
                    "gate",
                    "process_refund",
                    "get_customer",
                    "identity verification",
                ],
            },
            "1.5": {
                "statement": (
                    "Apply Agent SDK hooks for tool call interception and data normalization"
                ),
                "description": (
                    "PostToolUse hooks for transforming tool results before the "
                    "model processes them. Hooks intercepting outgoing tool calls "
                    "to enforce compliance. Hooks vs prompt instructions for "
                    "deterministic guarantees."
                ),
                "keywords": [
                    "PostToolUse",
                    "hook",
                    "intercept",
                    "normalize",
                    "data normalization",
                    "compliance",
                    "SDK",
                    "tool call interception",
                    "Unix timestamp",
                    "ISO 8601",
                ],
            },
            "1.6": {
                "statement": (
                    "Design task decomposition strategies for complex workflows"
                ),
                "description": (
                    "Fixed sequential pipelines (prompt chaining) vs dynamic "
                    "adaptive decomposition. Per-file local analysis plus "
                    "cross-file integration pass. Adaptive investigation plans "
                    "that generate subtasks based on intermediate findings."
                ),
                "keywords": [
                    "prompt chaining",
                    "task decomposition",
                    "sequential pipeline",
                    "adaptive decomposition",
                    "per-file analysis",
                    "cross-file",
                    "subtasks",
                    "dynamic",
                ],
            },
            "1.7": {
                "statement": "Manage session state, resumption, and forking",
                "description": (
                    "Named session resumption using --resume. fork_session for "
                    "independent branches from a shared baseline. Informing the "
                    "agent about file changes on resume. Starting fresh with "
                    "structured summary vs resuming stale sessions."
                ),
                "keywords": [
                    "--resume",
                    "session resumption",
                    "fork_session",
                    "session state",
                    "stale context",
                    "branch",
                    "session name",
                ],
            },
        },
    },
    "2": {
        "domain": "Domain2: Tool Design & MCP Integration",
        "tasks": {
            "2.1": {
                "statement": (
                    "Design effective tool interfaces with clear descriptions "
                    "and boundaries"
                ),
                "description": (
                    "Tool descriptions as primary mechanism for LLM tool "
                    "selection. Minimal descriptions lead to unreliable "
                    "selection. Include input formats, example queries, edge "
                    "cases, boundary explanations. Overlapping descriptions "
                    "cause misrouting."
                ),
                "keywords": [
                    "tool description",
                    "tool selection",
                    "tool interface",
                    "ambiguous description",
                    "overlapping tools",
                    "input format",
                    "boundary",
                    "misrouting",
                ],
            },
            "2.2": {
                "statement": "Implement structured error responses for MCP tools",
                "description": (
                    "MCP isError flag pattern. Error categories: transient, "
                    "validation, business, permission. Retryable vs "
                    "non-retryable errors. Structured metadata including "
                    "errorCategory, isRetryable boolean."
                ),
                "keywords": [
                    "isError",
                    "MCP error",
                    "errorCategory",
                    "isRetryable",
                    "transient",
                    "validation error",
                    "permission error",
                    "structured error",
                    "retry",
                ],
            },
            "2.3": {
                "statement": (
                    "Distribute tools appropriately across agents and configure "
                    "tool choice"
                ),
                "description": (
                    "Too many tools degrades selection reliability. Scoped tool "
                    "access per role. tool_choice options: auto, any, forced "
                    "tool selection. Restricting subagent tool sets to their "
                    "specialization."
                ),
                "keywords": [
                    "tool_choice",
                    "tool distribution",
                    "scoped tools",
                    "tool_choice auto",
                    "tool_choice any",
                    "forced tool",
                    "tool overload",
                    "specialization",
                ],
            },
            "2.4": {
                "statement": (
                    "Integrate MCP servers into Claude Code and agent workflows"
                ),
                "description": (
                    "MCP server scoping: project-level .mcp.json vs user-level "
                    "~/.claude.json. Environment variable expansion for "
                    "credentials. MCP resources exposing content catalogs. "
                    "Community MCP servers vs custom implementations."
                ),
                "keywords": [
                    "MCP server",
                    ".mcp.json",
                    "~/.claude.json",
                    "MCP integration",
                    "environment variable",
                    "MCP resource",
                    "project-scoped",
                    "user-scoped",
                    "GITHUB_TOKEN",
                ],
            },
            "2.5": {
                "statement": (
                    "Select and apply built-in tools (Read, Write, Edit, Bash, "
                    "Grep, Glob) effectively"
                ),
                "description": (
                    "Grep for content search. Glob for file path pattern "
                    "matching. Read/Write for full file ops. Edit for targeted "
                    "modifications using unique text matching. When Edit fails, "
                    "use Read + Write fallback."
                ),
                "keywords": [
                    "Grep",
                    "Glob",
                    "Read",
                    "Write",
                    "Edit",
                    "Bash",
                    "built-in tool",
                    "file search",
                    "pattern matching",
                    "unique anchor",
                ],
            },
        },
    },
    "3": {
        "domain": "Domain3: Claude Code Configuration & Workflows",
        "tasks": {
            "3.1": {
                "statement": (
                    "Configure CLAUDE.md files with appropriate hierarchy, "
                    "scoping, and modular organization"
                ),
                "description": (
                    "CLAUDE.md hierarchy: user-level ~/.claude/CLAUDE.md, "
                    "project-level .claude/CLAUDE.md, directory-level. "
                    "@import syntax for modular references. .claude/rules/ "
                    "directory for topic-specific rule files."
                ),
                "keywords": [
                    "CLAUDE.md",
                    "@import",
                    ".claude/rules/",
                    "user-level config",
                    "project-level config",
                    "directory-level",
                    "modular",
                    "/memory",
                ],
            },
            "3.2": {
                "statement": "Create and configure custom slash commands and skills",
                "description": (
                    "Project-scoped commands in .claude/commands/. User-scoped "
                    "in ~/.claude/commands/. Skills in .claude/skills/ with "
                    "SKILL.md frontmatter. context: fork for isolated sub-agent. "
                    "allowed-tools and argument-hint frontmatter."
                ),
                "keywords": [
                    "slash command",
                    ".claude/commands/",
                    "skills",
                    ".claude/skills/",
                    "SKILL.md",
                    "context: fork",
                    "allowed-tools",
                    "argument-hint",
                    "frontmatter",
                ],
            },
            "3.3": {
                "statement": (
                    "Apply path-specific rules for conditional convention loading"
                ),
                "description": (
                    ".claude/rules/ files with YAML frontmatter paths fields "
                    "containing glob patterns. Rules load only when editing "
                    "matching files. Better than directory-level CLAUDE.md for "
                    "conventions spanning multiple directories."
                ),
                "keywords": [
                    "path-specific rules",
                    ".claude/rules/",
                    "paths frontmatter",
                    "glob pattern",
                    "conditional loading",
                    "YAML frontmatter",
                    "terraform/**",
                    "**/*.test.tsx",
                ],
            },
            "3.4": {
                "statement": "Determine when to use plan mode vs direct execution",
                "description": (
                    "Plan mode for complex tasks: large-scale changes, multiple "
                    "valid approaches, architectural decisions, multi-file "
                    "modifications. Direct execution for simple well-scoped "
                    "changes. Explore subagent for isolating verbose discovery."
                ),
                "keywords": [
                    "plan mode",
                    "direct execution",
                    "architectural decision",
                    "multi-file",
                    "Explore subagent",
                    "planning",
                    "scope",
                ],
            },
            "3.5": {
                "statement": (
                    "Apply iterative refinement techniques for progressive improvement"
                ),
                "description": (
                    "Concrete input/output examples for consistent "
                    "transformations. Test-driven iteration. Interview pattern "
                    "for surfacing design considerations. Single message for "
                    "interacting problems vs sequential for independent."
                ),
                "keywords": [
                    "iterative refinement",
                    "few-shot example",
                    "test-driven",
                    "interview pattern",
                    "input/output example",
                    "progressive improvement",
                    "edge case",
                ],
            },
            "3.6": {
                "statement": "Integrate Claude Code into CI/CD pipelines",
                "description": (
                    "-p flag for non-interactive mode. --output-format json "
                    "and --json-schema for structured output. CLAUDE.md for "
                    "project context in CI. Independent review instance more "
                    "effective than self-review."
                ),
                "keywords": [
                    "-p flag",
                    "--print",
                    "non-interactive",
                    "CI",
                    "CD",
                    "pipeline",
                    "--output-format json",
                    "--json-schema",
                    "PR comment",
                    "automated review",
                ],
            },
        },
    },
    "4": {
        "domain": "Domain4: Prompt Engineering & Structured Output",
        "tasks": {
            "4.1": {
                "statement": (
                    "Design prompts with explicit criteria to improve precision "
                    "and reduce false positives"
                ),
                "description": (
                    "Explicit criteria over vague instructions. General "
                    "instructions like 'be conservative' fail vs specific "
                    "categorical criteria. False positive rates erode developer "
                    "trust. Severity criteria with concrete code examples."
                ),
                "keywords": [
                    "explicit criteria",
                    "false positive",
                    "precision",
                    "prompt criteria",
                    "severity",
                    "conservative",
                    "high-confidence",
                    "developer trust",
                ],
            },
            "4.2": {
                "statement": (
                    "Apply few-shot prompting to improve output consistency and quality"
                ),
                "description": (
                    "Few-shot examples for consistently formatted actionable "
                    "output. Demonstrate ambiguous-case handling. Enable "
                    "generalization to novel patterns. Reduce hallucination in "
                    "extraction tasks."
                ),
                "keywords": [
                    "few-shot",
                    "few-shot prompting",
                    "example",
                    "output consistency",
                    "ambiguous case",
                    "generalization",
                    "hallucination",
                    "format example",
                ],
            },
            "4.3": {
                "statement": (
                    "Enforce structured output using tool use and JSON schemas"
                ),
                "description": (
                    "Tool use with JSON schemas for guaranteed schema-compliant "
                    "output. tool_choice auto vs any vs forced. Strict schemas "
                    "eliminate syntax errors but not semantic errors. Optional "
                    "vs required fields. Enum with other + detail pattern."
                ),
                "keywords": [
                    "JSON schema",
                    "structured output",
                    "tool_use",
                    "schema-compliant",
                    "required fields",
                    "optional fields",
                    "enum",
                    "semantic error",
                    "syntax error",
                ],
            },
            "4.4": {
                "statement": (
                    "Implement validation, retry, and feedback loops for "
                    "extraction quality"
                ),
                "description": (
                    "Retry-with-error-feedback. Limits of retry when info is "
                    "absent from source. Tracking detected_pattern for false "
                    "positive analysis. Self-correction with calculated_total "
                    "vs stated_total and conflict_detected booleans."
                ),
                "keywords": [
                    "retry",
                    "validation",
                    "feedback loop",
                    "detected_pattern",
                    "self-correction",
                    "calculated_total",
                    "conflict_detected",
                    "extraction quality",
                    "retry-with-error",
                ],
            },
            "4.5": {
                "statement": "Design efficient batch processing strategies",
                "description": (
                    "Message Batches API: 50% cost savings, 24-hour window, no "
                    "latency SLA. Appropriate for non-blocking latency-tolerant "
                    "workloads. Does not support multi-turn tool calling. "
                    "custom_id for correlating responses."
                ),
                "keywords": [
                    "batch",
                    "Message Batches API",
                    "custom_id",
                    "24-hour",
                    "50% cost",
                    "latency-tolerant",
                    "non-blocking",
                    "overnight",
                    "batch processing",
                ],
            },
            "4.6": {
                "statement": (
                    "Design multi-instance and multi-pass review architectures"
                ),
                "description": (
                    "Self-review limitations: model retains generation context. "
                    "Independent review instances more effective. Multi-pass: "
                    "per-file local passes plus cross-file integration passes. "
                    "Verification passes with confidence self-reporting."
                ),
                "keywords": [
                    "multi-instance",
                    "multi-pass review",
                    "independent review",
                    "self-review",
                    "per-file pass",
                    "integration pass",
                    "cross-file",
                    "confidence",
                    "review architecture",
                ],
            },
        },
    },
    "5": {
        "domain": "Domain5: Context Management & Reliability",
        "tasks": {
            "5.1": {
                "statement": (
                    "Manage conversation context to preserve critical information "
                    "across long interactions"
                ),
                "description": (
                    "Progressive summarization risks losing numerical values and "
                    "dates. Lost-in-the-middle effect. Tool results accumulating "
                    "tokens. Persistent case facts block. Trimming verbose tool "
                    "outputs."
                ),
                "keywords": [
                    "context management",
                    "progressive summarization",
                    "lost in the middle",
                    "case facts",
                    "tool result accumulation",
                    "token budget",
                    "context preservation",
                    "trim",
                ],
            },
            "5.2": {
                "statement": (
                    "Design effective escalation and ambiguity resolution patterns"
                ),
                "description": (
                    "Escalation triggers: customer requests human, policy gaps, "
                    "cannot make progress. Sentiment-based escalation unreliable. "
                    "Multiple customer matches require clarification. Honor "
                    "explicit escalation requests immediately."
                ),
                "keywords": [
                    "escalation",
                    "ambiguity",
                    "human handoff",
                    "policy exception",
                    "sentiment",
                    "confidence score",
                    "clarification",
                    "multiple matches",
                    "escalation trigger",
                ],
            },
            "5.3": {
                "statement": (
                    "Implement error propagation strategies across multi-agent systems"
                ),
                "description": (
                    "Structured error context enabling coordinator recovery. "
                    "Access failures vs valid empty results. Generic error "
                    "statuses hide context. Avoid silently suppressing errors "
                    "or terminating entire workflows on single failures."
                ),
                "keywords": [
                    "error propagation",
                    "structured error context",
                    "partial results",
                    "coordinator recovery",
                    "empty result",
                    "suppress error",
                    "failure type",
                    "alternative approach",
                ],
            },
            "5.4": {
                "statement": (
                    "Manage context effectively in large codebase exploration"
                ),
                "description": (
                    "Context degradation in extended sessions. Scratchpad files "
                    "for persisting key findings. Subagent delegation for verbose "
                    "exploration. Structured state persistence for crash recovery. "
                    "/compact command."
                ),
                "keywords": [
                    "large codebase",
                    "context degradation",
                    "scratchpad",
                    "/compact",
                    "crash recovery",
                    "manifest",
                    "state persistence",
                    "codebase exploration",
                ],
            },
            "5.5": {
                "statement": (
                    "Design human review workflows and confidence calibration"
                ),
                "description": (
                    "Aggregate accuracy metrics may mask poor performance on "
                    "specific types. Stratified random sampling. Field-level "
                    "confidence scores calibrated on labeled validation sets. "
                    "Routing low-confidence extractions to human review."
                ),
                "keywords": [
                    "human review",
                    "confidence calibration",
                    "stratified sampling",
                    "field-level confidence",
                    "validation set",
                    "review routing",
                    "accuracy by segment",
                    "labeled data",
                ],
            },
            "5.6": {
                "statement": (
                    "Preserve information provenance and handle uncertainty in "
                    "multi-source synthesis"
                ),
                "description": (
                    "Source attribution lost during summarization. Structured "
                    "claim-source mappings. Conflicting statistics annotated with "
                    "sources. Temporal data requires publication dates. Rendering "
                    "different content types appropriately."
                ),
                "keywords": [
                    "provenance",
                    "source attribution",
                    "claim-source mapping",
                    "conflicting sources",
                    "uncertainty",
                    "multi-source",
                    "temporal data",
                    "synthesis",
                    "publication date",
                ],
            },
        },
    },
}

TASK_TO_CUSTOM_CATEGORY = {
    "1.1": "agent_loops",
    "1.2": "multi_agent_orchestration",
    "1.3": "subagent_invocation",
    "1.4": "workflow_enforcement",
    "1.5": "sdk_hooks",
    "1.6": "task_decomposition",
    "1.7": "session_management",
    "2.1": "tool_interface_design",
    "2.2": "mcp_error_handling",
    "2.3": "tool_distribution_and_choice",
    "2.4": "mcp_integration",
    "2.5": "built_in_tool_usage",
    "3.1": "claude_md_configuration",
    "3.2": "commands_and_skills",
    "3.3": "path_specific_rules",
    "3.4": "plan_mode_vs_execution",
    "3.5": "iterative_refinement",
    "3.6": "ci_cd_integration",
    "4.1": "precision_prompting",
    "4.2": "few_shot_prompting",
    "4.3": "structured_output",
    "4.4": "validation_and_retry",
    "4.5": "batch_processing",
    "4.6": "multi_pass_review",
    "5.1": "context_preservation",
    "5.2": "escalation_and_ambiguity",
    "5.3": "error_propagation",
    "5.4": "large_codebase_context",
    "5.5": "human_review_and_calibration",
    "5.6": "provenance_and_uncertainty",
}