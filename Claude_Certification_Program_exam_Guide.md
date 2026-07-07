# Claude Certification Program Exam Guide

---

## Domain 1: Agentic Architecture & Orchestration

### Task Statement 1.1: Design and implement agentic loops for autonomous task execution

**Knowledge of:**
- The agentic loop lifecycle: sending requests to Claude, inspecting `stop_reason` (`"tool_use"` vs `"end_turn"`), executing requested tools, and returning results for the next iteration
- How tool results are appended to conversation history so the model can reason about the next action
- The distinction between model-driven decision-making (Claude reasons about which tool to call next based on context) and pre-configured decision trees or tool sequences

**Skills in:**
- Implementing agentic loop control flow that continues when `stop_reason` is `"tool_use"` and terminates when `stop_reason` is `"end_turn"`
- Adding tool results to conversation context between iterations so the model can incorporate new information into its reasoning
- Avoiding anti-patterns such as parsing natural language signals to determine loop termination, setting arbitrary iteration caps as the primary stopping mechanism, or checking for assistant text content as a completion indicator

---

### Task Statement 1.2: Orchestrate multi-agent systems with coordinator-subagent patterns

**Knowledge of:**
- Hub-and-spoke architecture where a coordinator agent manages all inter-subagent communication, error handling, and information routing
- How subagents operate with isolated context—they do not inherit the coordinator's conversation history automatically
- The role of the coordinator in task decomposition, delegation, result aggregation, and deciding which subagents to invoke based on query complexity
- Risks of overly narrow task decomposition by the coordinator, leading to incomplete coverage of broad research topics

**Skills in:**
- Designing coordinator agents that analyze query requirements and dynamically select which subagents to invoke rather than always routing through the full pipeline
- Partitioning research scope across subagents to minimize duplication (e.g., assigning distinct subtopics or source types to each agent)
- Implementing iterative refinement loops where the coordinator evaluates synthesis output for gaps, re-delegates to search and analysis subagents with targeted queries, and re-invokes synthesis until coverage is sufficient
- Routing all subagent communication through the coordinator for observability, consistent error handling, and controlled information flow

---

### Task Statement 1.3: Configure subagent invocation, context passing, and spawning

**Knowledge of:**
- The Task tool as the mechanism for spawning subagents, and the requirement that `allowedTools` must include `"Task"` for a coordinator to invoke subagents
- That subagent context must be explicitly provided in the prompt—subagents do not automatically inherit parent context or share memory between invocations
- The `AgentDefinition` configuration including descriptions, system prompts, and tool restrictions for each subagent type
- Fork-based session management for exploring divergent approaches from a shared analysis baseline

**Skills in:**
- Including complete findings from prior agents directly in the subagent's prompt (e.g., passing web search results and document analysis outputs to the synthesis subagent)
- Using structured data formats to separate content from metadata (source URLs, document names, page numbers) when passing context between agents to preserve attribution
- Spawning parallel subagents by emitting multiple Task tool calls in a single coordinator response rather than across separate turns
- Designing coordinator prompts that specify research goals and quality criteria rather than step-by-step procedural instructions, to enable subagent adaptability

---

### Task Statement 1.4: Implement multi-step workflows with enforcement and handoff patterns

**Knowledge of:**
- The difference between programmatic enforcement (hooks, prerequisite gates) and prompt-based guidance for workflow ordering
- When deterministic compliance is required (e.g., identity verification before financial operations), prompt instructions alone have a non-zero failure rate
- Structured handoff protocols for mid-process escalation that include customer details, root cause analysis, and recommended actions

**Skills in:**
- Implementing programmatic prerequisites that block downstream tool calls until prerequisite steps have completed (e.g., blocking `process_refund` until `get_customer` has returned a verified customer ID)
- Decomposing multi-concern customer requests into distinct items, then investigating each in parallel using shared context before synthesizing a unified resolution
- Compiling structured handoff summaries (customer ID, root cause, refund amount, recommended action) when escalating to human agents who lack access to the conversation transcript

---

### Task Statement 1.5: Apply Agent SDK hooks for tool call interception and data normalization

**Knowledge of:**
- Hook patterns (e.g., `PostToolUse`) that intercept tool results for transformation before the model processes them
- Hook patterns that intercept outgoing tool calls to enforce compliance rules (e.g., blocking refunds above a threshold)
- The distinction between using hooks for deterministic guarantees versus relying on prompt instructions for probabilistic compliance

**Skills in:**
- Implementing `PostToolUse` hooks to normalize heterogeneous data formats (Unix timestamps, ISO 8601, numeric status codes) from different MCP tools before the agent processes them
- Implementing tool call interception hooks that block policy-violating actions (e.g., refunds exceeding $500) and redirect to alternative workflows (e.g., human escalation)
- Choosing hooks over prompt-based enforcement when business rules require guaranteed compliance

---

### Task Statement 1.6: Design task decomposition strategies for complex workflows

**Knowledge of:**
- When to use fixed sequential pipelines (prompt chaining) versus dynamic adaptive decomposition based on intermediate findings
- Prompt chaining patterns that break reviews into sequential steps (e.g., analyze each file individually, then run a cross-file integration pass)
- The value of adaptive investigation plans that generate subtasks based on what is discovered at each step

**Skills in:**
- Selecting task decomposition patterns appropriate to the workflow: prompt chaining for predictable multi-aspect reviews, dynamic decomposition for open-ended investigation tasks
- Splitting large code reviews into per-file local analysis passes plus a separate cross-file integration pass to avoid attention dilution
- Decomposing open-ended tasks (e.g., "add comprehensive tests to a legacy codebase") by first mapping structure, identifying high-impact areas, then creating a prioritized plan that adapts as dependencies are discovered

---

### Task Statement 1.7: Manage session state, resumption, and forking

**Knowledge of:**
- Named session resumption using `--resume <session-name>` to continue a specific prior conversation
- `fork_session` for creating independent branches from a shared analysis baseline to explore divergent approaches
- The importance of informing the agent about changes to previously analyzed files when resuming sessions after code modifications
- Why starting a new session with a structured summary is more reliable than resuming with stale tool results

**Skills in:**
- Using `--resume` with session names to continue named investigation sessions across work sessions
- Using `fork_session` to create parallel exploration branches (e.g., comparing two testing strategies or refactoring approaches from a shared codebase analysis)
- Choosing between session resumption (when prior context is mostly valid) and starting fresh with injected summaries (when prior tool results are stale)
- Informing a resumed session about specific file changes for targeted re-analysis rather than requiring full re-exploration

---

## Domain 2: Tool Design & MCP Integration

### Task Statement 2.1: Design effective tool interfaces with clear descriptions and boundaries

**Knowledge of:**
- Tool descriptions as the primary mechanism LLMs use for tool selection; minimal descriptions lead to unreliable selection among similar tools
- The importance of including input formats, example queries, edge cases, and boundary explanations in tool descriptions
- How ambiguous or overlapping tool descriptions cause misrouting (e.g., `analyze_content` vs `analyze_document` with near-identical descriptions)
- The impact of system prompt wording on tool selection: keyword-sensitive instructions can create unintended tool associations

**Skills in:**
- Writing tool descriptions that clearly differentiate each tool's purpose, expected inputs, outputs, and when to use it versus similar alternatives
- Renaming tools and updating descriptions to eliminate functional overlap (e.g., renaming `analyze_content` to `extract_web_results` with a web-specific description)
- Splitting generic tools into purpose-specific tools with defined input/output contracts (e.g., splitting a generic `analyze_document` into `extract_data_points`, `summarize_content`, and `verify_claim_against_source`)
- Reviewing system prompts for keyword-sensitive instructions that might override well-written tool descriptions

---

### Task Statement 2.2: Implement structured error responses for MCP tools

**Knowledge of:**
- The MCP `isError` flag pattern for communicating tool failures back to the agent
- The distinction between transient errors (timeouts, service unavailability), validation errors (invalid input), business errors (policy violations), and permission errors
- Why uniform error responses (generic "Operation failed") prevent the agent from making appropriate recovery decisions
- The difference between retryable and non-retryable errors, and how returning structured metadata prevents wasted retry attempts

**Skills in:**
- Returning structured error metadata including `errorCategory` (transient/validation/permission), `isRetryable` boolean, and human-readable descriptions
- Including `retriable: false` flags and customer-friendly explanations for business rule violations so the agent can communicate appropriately
- Implementing local error recovery within subagents for transient failures, propagating to the coordinator only errors that cannot be resolved locally along with partial results and what was attempted
- Distinguishing between access failures (needing retry decisions) and valid empty results (representing successful queries with no matches)

---

### Task Statement 2.3: Distribute tools appropriately across agents and configure tool choice

**Knowledge of:**
- The principle that giving an agent access to too many tools (e.g., 18 instead of 4–5) degrades tool selection reliability by increasing decision complexity
- Why agents with tools outside their specialization tend to misuse them (e.g., a synthesis agent attempting web searches)
- Scoped tool access: giving agents only the tools needed for their role, with limited cross-role tools for specific high-frequency needs
- `tool_choice` configuration options: `"auto"`, `"any"`, and forced tool selection (`{"type": "tool", "name": "..."}`)

**Skills in:**
- Restricting each subagent's tool set to those relevant to its role, preventing cross-specialization misuse
- Replacing generic tools with constrained alternatives (e.g., replacing `fetch_url` with `load_document` that validates document URLs)
- Providing scoped cross-role tools for high-frequency needs (e.g., a `verify_fact` tool for the synthesis agent) while routing complex cases through the coordinator
- Using `tool_choice` forced selection to ensure a specific tool is called first (e.g., forcing `extract_metadata` before enrichment tools), then processing subsequent steps in follow-up turns
- Setting `tool_choice: "any"` to guarantee the model calls a tool rather than returning conversational text

---

### Task Statement 2.4: Integrate MCP servers into Claude Code and agent workflows

**Knowledge of:**
- MCP server scoping: project-level (`.mcp.json`) for shared team tooling vs user-level (`~/.claude.json`) for personal/experimental servers
- Environment variable expansion in `.mcp.json` (e.g., `${GITHUB_TOKEN}`) for credential management without committing secrets
- That tools from all configured MCP servers are discovered at connection time and available simultaneously to the agent
- MCP resources as a mechanism for exposing content catalogs (e.g., issue summaries, documentation hierarchies, database schemas) to reduce exploratory tool calls

**Skills in:**
- Configuring shared MCP servers in project-scoped `.mcp.json` with environment variable expansion for authentication tokens
- Configuring personal/experimental MCP servers in user-scoped `~/.claude.json`
- Enhancing MCP tool descriptions to explain capabilities and outputs in detail, preventing the agent from preferring built-in tools (like Grep) over more capable MCP tools
- Choosing existing community MCP servers over custom implementations for standard integrations (e.g., Jira), reserving custom servers for team-specific workflows
- Exposing content catalogs as MCP resources to give agents visibility into available data without requiring exploratory tool calls

---

### Task Statement 2.5: Select and apply built-in tools (Read, Write, Edit, Bash, Grep, Glob) effectively

**Knowledge of:**
- `Grep` for content search (searching file contents for patterns like function names, error messages, or import statements)
- `Glob` for file path pattern matching (finding files by name or extension patterns)
- `Read`/`Write` for full file operations; `Edit` for targeted modifications using unique text matching
- When `Edit` fails due to non-unique text matches, using `Read` + `Write` as a fallback for reliable file modifications

**Skills in:**
- Selecting `Grep` for searching code content across a codebase (e.g., finding all callers of a function, locating error messages)
- Selecting `Glob` for finding files matching naming patterns (e.g., `**/*.test.tsx`)
- Using `Read` to load full file contents followed by `Write` when `Edit` cannot find unique anchor text
- Building codebase understanding incrementally: starting with `Grep` to find entry points, then using `Read` to follow imports and trace flows, rather than reading all files upfront
- Tracing function usage across wrapper modules by first identifying all exported names, then searching for each name across the codebase

---

## Domain 3: Claude Code Configuration & Workflows

### Task Statement 3.1: Configure CLAUDE.md files with appropriate hierarchy, scoping, and modular organization

**Knowledge of:**
- The `CLAUDE.md` configuration hierarchy: user-level (`~/.claude/CLAUDE.md`), project-level (`.claude/CLAUDE.md` or root `CLAUDE.md`), and directory-level (subdirectory `CLAUDE.md` files)
- That user-level settings apply only to that user—instructions in `~/.claude/CLAUDE.md` are not shared with teammates via version control
- The `@import` syntax for referencing external files to keep `CLAUDE.md` modular (e.g., importing specific standards files relevant to each package)
- `.claude/rules/` directory for organizing topic-specific rule files as an alternative to a monolithic `CLAUDE.md`

**Skills in:**
- Diagnosing configuration hierarchy issues (e.g., a new team member not receiving instructions because they're in user-level rather than project-level configuration)
- Using `@import` to selectively include relevant standards files in each package's `CLAUDE.md` based on maintainer domain knowledge
- Splitting large `CLAUDE.md` files into focused topic-specific files in `.claude/rules/` (e.g., `testing.md`, `api-conventions.md`, `deployment.md`)
- Using the `/memory` command to verify which memory files are loaded and diagnose inconsistent behavior across sessions

---

### Task Statement 3.2: Create and configure custom slash commands and skills

**Knowledge of:**
- Project-scoped commands in `.claude/commands/` (shared via version control) vs user-scoped commands in `~/.claude/commands/` (personal)
- Skills in `.claude/skills/` with `SKILL.md` files that support frontmatter configuration including `context: fork`, `allowed-tools`, and `argument-hint`
- The `context: fork` frontmatter option for running skills in an isolated sub-agent context, preventing skill outputs from polluting the main conversation
- Personal skill customization: creating personal variants in `~/.claude/skills/` with different names to avoid affecting teammates

**Skills in:**
- Creating project-scoped slash commands in `.claude/commands/` for team-wide availability via version control
- Using `context: fork` to isolate skills that produce verbose output (e.g., codebase analysis) or exploratory context (e.g., brainstorming alternatives) from the main session
- Configuring `allowed-tools` in skill frontmatter to restrict tool access during skill execution (e.g., limiting to file write operations to prevent destructive actions)
- Using `argument-hint` frontmatter to prompt developers for required parameters when they invoke the skill without arguments
- Choosing between skills (on-demand invocation for task-specific workflows) and `CLAUDE.md` (always-loaded universal standards)

---

### Task Statement 3.3: Apply path-specific rules for conditional convention loading

**Knowledge of:**
- `.claude/rules/` files with YAML frontmatter `paths` fields containing glob patterns for conditional rule activation
- How path-scoped rules load only when editing matching files, reducing irrelevant context and token usage
- The advantage of glob-pattern rules over directory-level `CLAUDE.md` files for conventions that span multiple directories (e.g., test files spread throughout a codebase)

**Skills in:**
- Creating `.claude/rules/` files with YAML frontmatter path scoping (e.g., `paths: ["terraform/**/*"]`) so rules load only when editing matching files
- Using glob patterns in path-specific rules to apply conventions to files by type regardless of directory location (e.g., `**/*.test.tsx` for all test files)
- Choosing path-specific rules over subdirectory `CLAUDE.md` files when conventions must apply to files spread across the codebase

---

### Task Statement 3.4: Determine when to use plan mode vs direct execution

**Knowledge of:**
- Plan mode is designed for complex tasks involving large-scale changes, multiple valid approaches, architectural decisions, and multi-file modifications
- Direct execution is appropriate for simple, well-scoped changes (e.g., adding a single validation check to one function)
- Plan mode enables safe codebase exploration and design before committing to changes, preventing costly rework
- The Explore subagent for isolating verbose discovery output and returning summaries to preserve main conversation context

**Skills in:**
- Selecting plan mode for tasks with architectural implications (e.g., microservice restructuring, library migrations affecting 45+ files, choosing between integration approaches with different infrastructure requirements)
- Selecting direct execution for well-understood changes with clear scope (e.g., a single-file bug fix with a clear stack trace, adding a date validation conditional)
- Using the Explore subagent for verbose discovery phases to prevent context window exhaustion during multi-phase tasks
- Combining plan mode for investigation with direct execution for implementation (e.g., planning a library migration, then executing the planned approach)

---

### Task Statement 3.5: Apply iterative refinement techniques for progressive improvement

**Knowledge of:**
- Concrete input/output examples as the most effective way to communicate expected transformations when prose descriptions are interpreted inconsistently
- Test-driven iteration: writing test suites first, then iterating by sharing test failures to guide progressive improvement
- The interview pattern: having Claude ask questions to surface considerations the developer may not have anticipated before implementing
- When to provide all issues in a single message (interacting problems) versus fixing them sequentially (independent problems)

**Skills in:**
- Providing 2–3 concrete input/output examples to clarify transformation requirements when natural language descriptions produce inconsistent results
- Writing test suites covering expected behavior, edge cases, and performance requirements before implementation, then iterating by sharing test failures
- Using the interview pattern to surface design considerations (e.g., cache invalidation strategies, failure modes) before implementing solutions in unfamiliar domains
- Providing specific test cases with example input and expected output to fix edge case handling (e.g., null values in migration scripts)
- Addressing multiple interacting issues in a single detailed message when fixes interact, versus sequential iteration for independent issues

---

### Task Statement 3.6: Integrate Claude Code into CI/CD pipelines

**Knowledge of:**
- The `-p` (or `--print`) flag for running Claude Code in non-interactive mode in automated pipelines
- `--output-format json` and `--json-schema` CLI flags for enforcing structured output in CI contexts
- `CLAUDE.md` as the mechanism for providing project context (testing standards, fixture conventions, review criteria) to CI-invoked Claude Code
- Session context isolation: why the same Claude session that generated code is less effective at reviewing its own changes compared to an independent review instance

**Skills in:**
- Running Claude Code in CI with the `-p` flag to prevent interactive input hangs
- Using `--output-format json` with `--json-schema` to produce machine-parseable structured findings for automated posting as inline PR comments
- Including prior review findings in context when re-running reviews after new commits, instructing Claude to report only new or still-unaddressed issues to avoid duplicate comments
- Providing existing test files in context so test generation avoids suggesting duplicate scenarios already covered by the test suite
- Documenting testing standards, valuable test criteria, and available fixtures in `CLAUDE.md` to improve test generation quality and reduce low-value test output

---

## Domain 4: Prompt Engineering & Structured Output

### Task Statement 4.1: Design prompts with explicit criteria to improve precision and reduce false positives

**Knowledge of:**
- The importance of explicit criteria over vague instructions (e.g., "flag comments only when claimed behavior contradicts actual code behavior" vs "check that comments are accurate")
- How general instructions like "be conservative" or "only report high-confidence findings" fail to improve precision compared to specific categorical criteria
- The impact of false positive rates on developer trust: high false positive categories undermine confidence in accurate categories

**Skills in:**
- Writing specific review criteria that define which issues to report (bugs, security) versus skip (minor style, local patterns) rather than relying on confidence-based filtering
- Temporarily disabling high false-positive categories to restore developer trust while improving prompts for those categories
- Defining explicit severity criteria with concrete code examples for each severity level to achieve consistent classification

---

### Task Statement 4.2: Apply few-shot prompting to improve output consistency and quality

**Knowledge of:**
- Few-shot examples as the most effective technique for achieving consistently formatted, actionable output when detailed instructions alone produce inconsistent results
- The role of few-shot examples in demonstrating ambiguous-case handling (e.g., tool selection for ambiguous requests, branch-level test coverage gaps)
- How few-shot examples enable the model to generalize judgment to novel patterns rather than matching only pre-specified cases
- The effectiveness of few-shot examples for reducing hallucination in extraction tasks (e.g., handling informal measurements, varied document structures)

**Skills in:**
- Creating 2–4 targeted few-shot examples for ambiguous scenarios that show reasoning for why one action was chosen over plausible alternatives
- Including few-shot examples that demonstrate specific desired output format (location, issue, severity, suggested fix) to achieve consistency
- Providing few-shot examples distinguishing acceptable code patterns from genuine issues to reduce false positives while enabling generalization
- Using few-shot examples to demonstrate correct handling of varied document structures (inline citations vs bibliographies, methodology sections vs embedded details)
- Adding few-shot examples showing correct extraction from documents with varied formats to address empty/null extraction of required fields

---

### Task Statement 4.3: Enforce structured output using tool use and JSON schemas

**Knowledge of:**
- Tool use (`tool_use`) with JSON schemas as the most reliable approach for guaranteed schema-compliant structured output, eliminating JSON syntax errors
- The distinction between `tool_choice: "auto"` (model may return text instead of calling a tool), `"any"` (model must call a tool but can choose which), and forced tool selection (model must call a specific named tool)
- That strict JSON schemas via tool use eliminate syntax errors but do not prevent semantic errors (e.g., line items that don't sum to total, values in wrong fields)
- Schema design considerations: required vs optional fields, enum fields with `"other"` + detail string patterns for extensible categories

**Skills in:**
- Defining extraction tools with JSON schemas as input parameters and extracting structured data from the `tool_use` response
- Setting `tool_choice: "any"` to guarantee structured output when multiple extraction schemas exist and the document type is unknown
- Forcing a specific tool with `tool_choice: {"type": "tool", "name": "extract_metadata"}` to ensure a particular extraction runs before enrichment steps
- Designing schema fields as optional (nullable) when source documents may not contain the information, preventing the model from fabricating values to satisfy required fields
- Adding enum values like `"unclear"` for ambiguous cases and `"other"` + detail fields for extensible categorization
- Including format normalization rules in prompts alongside strict output schemas to handle inconsistent source formatting

---

### Task Statement 4.4: Implement validation, retry, and feedback loops for extraction quality

**Knowledge of:**
- Retry-with-error-feedback: appending specific validation errors to the prompt on retry to guide the model toward correction
- The limits of retry: retries are ineffective when the required information is simply absent from the source document (vs format or structural errors)
- Feedback loop design: tracking which code constructs trigger findings (`detected_pattern` field) to enable systematic analysis of dismissal patterns
- The difference between semantic validation errors (values don't sum, wrong field placement) and schema syntax errors (eliminated by tool use)

**Skills in:**
- Implementing follow-up requests that include the original document, the failed extraction, and specific validation errors for model self-correction
- Identifying when retries will be ineffective (e.g., information exists only in an external document not provided) versus when they will succeed (format mismatches, structural output errors)
- Adding `detected_pattern` fields to structured findings to enable analysis of false positive patterns when developers dismiss findings
- Designing self-correction validation flows: extracting `"calculated_total"` alongside `"stated_total"` to flag discrepancies, adding `"conflict_detected"` booleans for inconsistent source data

---

### Task Statement 4.5: Design efficient batch processing strategies

**Knowledge of:**
- The Message Batches API: 50% cost savings, up to 24-hour processing window, no guaranteed latency SLA
- Batch processing is appropriate for non-blocking, latency-tolerant workloads (overnight reports, weekly audits, nightly test generation) and inappropriate for blocking workflows (pre-merge checks)
- The batch API does not support multi-turn tool calling within a single request (cannot execute tools mid-request and return results)
- `custom_id` fields for correlating batch request/response pairs

**Skills in:**
- Matching API approach to workflow latency requirements: synchronous API for blocking pre-merge checks, batch API for overnight/weekly analysis
- Calculating batch submission frequency based on SLA constraints (e.g., 4-hour windows to guarantee 30-hour SLA with 24-hour batch processing)
- Handling batch failures: resubmitting only failed documents (identified by `custom_id`) with appropriate modifications (e.g., chunking documents that exceeded context limits)
- Using prompt refinement on a sample set before batch-processing large volumes to maximize first-pass success rates and reduce iterative resubmission costs

---

### Task Statement 4.6: Design multi-instance and multi-pass review architectures

**Knowledge of:**
- Self-review limitations: a model retains reasoning context from generation, making it less likely to question its own decisions in the same session
- Independent review instances (without prior reasoning context) are more effective at catching subtle issues than self-review instructions or extended thinking
- Multi-pass review: splitting large reviews into per-file local analysis passes plus cross-file integration passes to avoid attention dilution and contradictory findings

**Skills in:**
- Using a second independent Claude instance to review generated code without the generator's reasoning context
- Splitting large multi-file reviews into focused per-file passes for local issues plus separate integration passes for cross-file data flow analysis
- Running verification passes where the model self-reports confidence alongside each finding to enable calibrated review routing

---

## Domain 5: Context Management & Reliability

### Task Statement 5.1: Manage conversation context to preserve critical information across long interactions

**Knowledge of:**
- Progressive summarization risks: condensing numerical values, percentages, dates, and customer-stated expectations into vague summaries
- The "lost in the middle" effect: models reliably process information at the beginning and end of long inputs but may omit findings from middle sections
- How tool results accumulate in context and consume tokens disproportionately to their relevance (e.g., 40+ fields per order lookup when only 5 are relevant)
- The importance of passing complete conversation history in subsequent API requests to maintain conversational coherence

**Skills in:**
- Extracting transactional facts (amounts, dates, order numbers, statuses) into a persistent "case facts" block included in each prompt, outside summarized history
- Extracting and persisting structured issue data (order IDs, amounts, statuses) into a separate context layer for multi-issue sessions
- Trimming verbose tool outputs to only relevant fields before they accumulate in context (e.g., keeping only return-relevant fields from order lookups)
- Placing key findings summaries at the beginning of aggregated inputs and organizing detailed results with explicit section headers to mitigate position effects
- Requiring subagents to include metadata (dates, source locations, methodological context) in structured outputs to support accurate downstream synthesis
- Modifying upstream agents to return structured data (key facts, citations, relevance scores) instead of verbose content and reasoning chains when downstream agents have limited context budgets

---

### Task Statement 5.2: Design effective escalation and ambiguity resolution patterns

**Knowledge of:**
- Appropriate escalation triggers: customer requests for a human, policy exceptions/gaps (not just complex cases), and inability to make meaningful progress
- The distinction between escalating immediately when a customer explicitly demands it versus offering to resolve when the issue is straightforward
- Why sentiment-based escalation and self-reported confidence scores are unreliable proxies for actual case complexity
- How multiple customer matches require clarification (requesting additional identifiers) rather than heuristic selection

**Skills in:**
- Adding explicit escalation criteria with few-shot examples to the system prompt demonstrating when to escalate versus resolve autonomously
- Honoring explicit customer requests for human agents immediately without first attempting investigation
- Acknowledging frustration while offering resolution when the issue is within the agent's capability, escalating only if the customer reiterates their preference
- Escalating when policy is ambiguous or silent on the customer's specific request (e.g., competitor price matching when policy only addresses own-site adjustments)
- Instructing the agent to ask for additional identifiers when tool results return multiple matches, rather than selecting based on heuristics

---

### Task Statement 5.3: Implement error propagation strategies across multi-agent systems

**Knowledge of:**
- Structured error context (failure type, attempted query, partial results, alternative approaches) as enabling intelligent coordinator recovery decisions
- The distinction between access failures (timeouts needing retry decisions) and valid empty results (successful queries with no matches)
- Why generic error statuses ("search unavailable") hide valuable context from the coordinator
- Why silently suppressing errors (returning empty results as success) or terminating entire workflows on single failures are both anti-patterns

**Skills in:**
- Returning structured error context including failure type, what was attempted, partial results, and potential alternatives to enable coordinator recovery
- Distinguishing access failures from valid empty results in error reporting so the coordinator can make appropriate decisions
- Having subagents implement local recovery for transient failures and only propagate errors they cannot resolve, including what was attempted and partial results
- Structuring synthesis output with coverage annotations indicating which findings are well-supported versus which topic areas have gaps due to unavailable sources

---

### Task Statement 5.4: Manage context effectively in large codebase exploration

**Knowledge of:**
- Context degradation in extended sessions: models start giving inconsistent answers and referencing "typical patterns" rather than specific classes discovered earlier
- The role of scratchpad files for persisting key findings across context boundaries
- Subagent delegation for isolating verbose exploration output while the main agent coordinates high-level understanding
- Structured state persistence for crash recovery: each agent exports state to a known location, and the coordinator loads a manifest on resume

**Skills in:**
- Spawning subagents to investigate specific questions (e.g., "find all test files," "trace refund flow dependencies") while the main agent preserves high-level coordination
- Having agents maintain scratchpad files recording key findings, referencing them for subsequent questions to counteract context degradation
- Summarizing key findings from one exploration phase before spawning sub-agents for the next phase, injecting summaries into initial context
- Designing crash recovery using structured agent state exports (manifests) that the coordinator loads on resume and injects into agent prompts
- Using `/compact` to reduce context usage during extended exploration sessions when context fills with verbose discovery output

---

### Task Statement 5.5: Design human review workflows and confidence calibration

**Knowledge of:**
- The risk that aggregate accuracy metrics (e.g., 97% overall) may mask poor performance on specific document types or fields
- Stratified random sampling for measuring error rates in high-confidence extractions and detecting novel error patterns
- Field-level confidence scores calibrated using labeled validation sets for routing review attention
- The importance of validating accuracy by document type and field segment before automating high-confidence extractions

**Skills in:**
- Implementing stratified random sampling of high-confidence extractions for ongoing error rate measurement and novel pattern detection
- Analyzing accuracy by document type and field to verify consistent performance across all segments before reducing human review
- Having models output field-level confidence scores, then calibrating review thresholds using labeled validation sets
- Routing extractions with low model confidence or ambiguous/contradictory source documents to human review, prioritizing limited reviewer capacity

---

### Task Statement 5.6: Preserve information provenance and handle uncertainty in multi-source synthesis

**Knowledge of:**
- How source attribution is lost during summarization steps when findings are compressed without preserving claim-source mappings
- The importance of structured claim-source mappings that the synthesis agent must preserve and merge when combining findings
- How to handle conflicting statistics from credible sources: annotating conflicts with source attribution rather than arbitrarily selecting one value
- Temporal data: requiring publication/collection dates in structured outputs to prevent temporal differences from being misinterpreted as contradictions

**Skills in:**
- Requiring subagents to output structured claim-source mappings (source URLs, document names, relevant excerpts) that downstream agents preserve through synthesis
- Structuring reports with explicit sections distinguishing well-established findings from contested ones, preserving original source characterizations and methodological context
- Completing document analysis with conflicting values included and explicitly annotated, letting the coordinator decide how to reconcile before passing to synthesis
- Requiring subagents to include publication or data collection dates in structured outputs to enable correct temporal interpretation
- Rendering different content types appropriately in synthesis outputs—financial data as tables, news as prose, technical findings as structured lists—rather than converting everything to a uniform format