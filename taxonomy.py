TAXONOMY = {
    "1": {
        "domain": "Domain1: Agentic Architecture & Orchestration",
        "tasks": {
            "1.1": (
                "Design and implement agentic loops for autonomous task "
                "execution"
            ),
            "1.2": (
                "Orchestrate multi-agent systems with coordinator-subagent "
                "patterns"
            ),
            "1.3": (
                "Configure subagent invocation, context passing, and spawning"
            ),
            "1.4": (
                "Implement multi-step workflows with enforcement and handoff "
                "patterns"
            ),
            "1.5": (
                "Apply Agent SDK hooks for tool call interception and data "
                "normalization"
            ),
            "1.6": (
                "Design task decomposition strategies for complex workflows"
            ),
            "1.7": "Manage session state, resumption, and forking",
        },
    },
    "2": {
        "domain": "Domain2: Tool Design & MCP Integration",
        "tasks": {
            "2.1": (
                "Design effective tool interfaces with clear descriptions and "
                "boundaries"
            ),
            "2.2": "Implement structured error responses for MCP tools",
            "2.3": (
                "Distribute tools appropriately across agents and configure "
                "tool choice"
            ),
            "2.4": (
                "Integrate MCP servers into Claude Code and agent workflows"
            ),
            "2.5": (
                "Select and apply built-in tools (Read, Write, Edit, Bash, "
                "Grep, Glob) effectively"
            ),
        },
    },
    "3": {
        "domain": "Domain3: Claude Code Configuration & Workflows",
        "tasks": {
            "3.1": (
                "Configure CLAUDE.md files with appropriate hierarchy, "
                "scoping, and modular organization"
            ),
            "3.2": "Create and configure custom slash commands and skills",
            "3.3": (
                "Apply path-specific rules for conditional convention loading"
            ),
            "3.4": "Determine when to use plan mode vs direct execution",
            "3.5": (
                "Apply iterative refinement techniques for progressive "
                "improvement"
            ),
            "3.6": "Integrate Claude Code into CI/CD pipelines",
        },
    },
    "4": {
        "domain": "Domain4: Prompt Engineering & Structured Output",
        "tasks": {
            "4.1": (
                "Design prompts with explicit criteria to improve precision "
                "and reduce false positives"
            ),
            "4.2": (
                "Apply few-shot prompting to improve output consistency and "
                "quality"
            ),
            "4.3": (
                "Enforce structured output using tool use and JSON schemas"
            ),
            "4.4": (
                "Implement validation, retry, and feedback loops for "
                "extraction quality"
            ),
            "4.5": "Design efficient batch processing strategies",
            "4.6": (
                "Design multi-instance and multi-pass review architectures"
            ),
        },
    },
    "5": {
        "domain": "Domain5: Context Management & Reliability",
        "tasks": {
            "5.1": (
                "Manage conversation context to preserve critical information "
                "across long interactions"
            ),
            "5.2": (
                "Design effective escalation and ambiguity resolution patterns"
            ),
            "5.3": (
                "Implement error propagation strategies across multi-agent "
                "systems"
            ),
            "5.4": (
                "Manage context effectively in large codebase exploration"
            ),
            "5.5": (
                "Design human review workflows and confidence calibration"
            ),
            "5.6": (
                "Preserve information provenance and handle uncertainty in "
                "multi-source synthesis"
            ),
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