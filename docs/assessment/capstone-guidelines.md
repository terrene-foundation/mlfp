# Team Capstone Guidelines

**Weight**: 35% of final grade  
**Format**: Working production system + live demo + documentation  
**Team Size**: 3-4 members

---

## Overview

The team capstone is a production-grade ML system that demonstrates your team's ability to integrate multiple Kailash packages into a coherent, deployable solution. You choose the domain, define the problem, build the system, and present it live.

This is not a prototype or proof-of-concept. The expectation is a system that could be handed to an operations team for deployment. Data pipelines run reliably, models are versioned and monitored, the API serves predictions, and governance is documented.

---

## Requirements

### Kailash Package Integration

Your system must use at least 3 Kailash packages from the following:

| Package              | Purpose                | Example Usage                                              |
| -------------------- | ---------------------- | ---------------------------------------------------------- |
| `kailash` (Core SDK) | Workflow orchestration | Data processing pipelines, task automation                 |
| `kailash-ml`         | ML lifecycle           | Training, feature stores, drift monitoring, model registry |
| `kailash-dataflow`   | Database operations    | Data ingestion, storage, query pipelines                   |
| `kailash-nexus`      | Deployment             | API endpoints, CLI interface, MCP server                   |
| `kailash-kaizen`     | AI agents              | Autonomous decision-making, multi-agent coordination       |
| `kailash-pact`       | Governance             | D/T/R accountability, operating envelopes, audit trails    |
| `kailash-align`      | LLM alignment          | Fine-tuning, adapter management, model serving             |

### Mandatory System Components

Every capstone must include all four of the following:

#### 1. Data Pipeline (DataFlow)

- Automated data ingestion from at least one source
- Data validation and quality checks
- Schema management and versioning
- Error handling for malformed or missing data

#### 2. Model Lifecycle (kailash-ml)

- At least one trained model registered in `ModelRegistry`
- Feature management through `FeatureStore`
- Hyperparameter tuning via `HyperparameterSearch`
- Model versioning with metadata and lineage tracking

#### 3. Deployment (Nexus)

- Working API endpoint that serves predictions
- Input validation on the API
- Health check endpoint
- At least one additional channel (CLI or MCP) beyond the API

#### 4. Governance OR Agents (choose one)

**Option A — Governance (PACT)**:

- D/T/R (Decision/Trust/Responsibility) specification for your system
- Operating envelope definition (what the system is and is not authorised to do)
- Audit trail for model decisions
- Documented escalation paths

**Option B — Agents (Kaizen)**:

- At least one autonomous agent that performs a meaningful task in your system
- Agent safety constraints (what the agent cannot do)
- Human-in-the-loop checkpoints where appropriate
- Agent decision logging

---

## Presentation

### Live Demo (15 minutes)

Your team presents a live demonstration of the working system:

- **System overview** (2 min): Problem statement, architecture diagram, team member roles
- **Data pipeline** (3 min): Show data flowing from source through validation to storage
- **Model training and registry** (3 min): Demonstrate training, versioning, and model selection
- **Live predictions** (3 min): Send requests to the Nexus API, show responses, explain outputs
- **Governance/Agents** (2 min): Show PACT governance in action or agent decision-making
- **Monitoring** (2 min): Show drift monitoring dashboard, demonstrate an alert scenario

### Q&A (10 minutes)

Instructors and peers ask questions. Every team member must answer at least one question. Questions will cover:

- Architecture decisions ("Why did you choose X over Y?")
- Failure modes ("What happens when the data source goes down?")
- Statistical choices ("Why this model? What about the calibration?")
- Production concerns ("How would this scale to 10x the data?")

### Presentation Tips

- Rehearse the demo end-to-end at least twice before the session
- Have a backup plan if the live API fails (pre-recorded demo as fallback, not a substitute)
- Every team member should present at least one section
- Avoid slides with code dumps — show the running system instead

---

## Deliverables

All deliverables are submitted via the team's GitHub repository.

### 1. Working System

```
capstone/
    README.md                  # Setup instructions, architecture overview
    pyproject.toml             # All dependencies pinned
    src/
        pipeline/              # DataFlow data pipeline
        features/              # FeatureStore and FeatureEngineer
        training/              # TrainingPipeline and model training
        serving/               # Nexus API configuration
        governance/            # PACT specs (if Option A)
        agents/                # Kaizen agents (if Option B)
        monitoring/            # DriftMonitor configuration
    tests/
        test_pipeline.py       # Data pipeline tests
        test_model.py          # Model training tests
        test_api.py            # API endpoint tests
    scripts/
        setup.sh               # One-command environment setup
        train.sh               # One-command model training
        serve.sh               # One-command API launch
```

The system must be runnable from a clean environment following the README instructions.

### 2. Architecture Document

A 3-5 page document covering:

- System architecture diagram (showing all Kailash packages and how they connect)
- Data flow diagram (source to prediction)
- Technology choices and justifications
- Scalability considerations
- Security considerations (no hardcoded secrets, input validation)

### 3. Model Cards

One model card per trained model (use the template at `docs/assessment/model-card-template.md`):

- All sections completed with substantive content
- Fairness analysis if applicable to your domain
- Honest limitations

### 4. Governance Specification (if Option A)

- D/T/R matrix for all system decisions
- Operating envelope document
- Audit log sample (showing 10+ logged decisions)
- Escalation procedure document

### 5. Agent Specification (if Option B)

- Agent capability description and safety constraints
- Decision log sample (showing 10+ agent decisions)
- Human-in-the-loop checkpoint documentation
- Failure mode analysis (what happens when the agent makes a bad decision?)

---

## Timeline

| Week  | Milestone     | Deliverable                                                        | Review                                             |
| ----- | ------------- | ------------------------------------------------------------------ | -------------------------------------------------- |
| **2** | Proposal      | 1-page proposal: problem, dataset, architecture sketch, team roles | Instructor feedback (not graded)                   |
| **3** | Data pipeline | Working DataFlow pipeline with data validation                     | Self-check against rubric                          |
| **4** | Checkpoint    | Model trained, registered, basic API serving predictions           | Instructor checkpoint (5 min per team, not graded) |
| **5** | Integration   | All components connected, governance/agents functional             | Team rehearsal                                     |
| **6** | Submission    | Final repository + live demo                                       | Graded presentation + peer review                  |

### Proposal Requirements (Week 2)

Your 1-page proposal must include:

- **Problem statement**: What are you building and why?
- **Dataset**: What data will you use? (Must be publicly available or synthetic with documented generation)
- **Architecture sketch**: Which Kailash packages, how they connect
- **Team roles**: Who is responsible for which component (every member must own at least one component)
- **Risk assessment**: What could go wrong and how will you mitigate it?

### Checkpoint Requirements (Week 4)

At the checkpoint, demonstrate:

- Data pipeline ingests and validates data
- At least one model is trained and registered in `ModelRegistry`
- Basic Nexus API returns predictions (even if not fully polished)
- Team is on track for governance/agent integration

Teams that cannot demonstrate a working pipeline at the checkpoint will receive a mandatory meeting with the instructor to create a recovery plan.

---

## Team Formation

- Teams are self-formed (3-4 members)
- Each team member must own at least one major component (pipeline, model, deployment, governance/agents)
- Teams submit a contribution log with their final deliverable showing each member's commits and responsibilities
- Individual grades may be adjusted based on contribution evidence (see Teamwork criterion in rubric)

### Contribution Log Format

Include in your repository as `CONTRIBUTIONS.md`:

```markdown
# Team Contributions

## Member A (Name)

- Primary: Data pipeline (DataFlow)
- Secondary: Data validation, schema design
- Key commits: [list 5-10 significant commits with descriptions]

## Member B (Name)

- Primary: Model training and registry (kailash-ml)
- Secondary: Feature engineering
- Key commits: [list 5-10 significant commits with descriptions]

...
```

---

## Grading

Capstones are graded against the rubric in `docs/assessment/capstone-rubric.md`. The five criteria are:

1. **System Architecture** (20%) — Clean separation, proper Kailash patterns, integration quality
2. **Technical Depth** (25%) — Appropriate ML techniques, statistical rigor, engineering quality
3. **Production Quality** (25%) — Governance, monitoring, deployment, reliability
4. **Presentation** (15%) — Demo quality, Q&A handling, communication clarity
5. **Teamwork** (15%) — Contribution equity, collaboration evidence, role clarity

---

## Academic Integrity

The capstone is a team effort. All code must be written by team members. You may use Kailash SDK documentation, course materials, and publicly available references with citation.

Teams must not share code with other teams. If two teams submit systems with substantially similar architecture or code, both teams will be investigated under the academic integrity policy.

---

## Frequently Asked Questions

**Can we use a private dataset from our employer?**  
Only if you have written permission and the data contains no PII or proprietary information. Prefer publicly available datasets to avoid complications.

**What if our live demo fails during the presentation?**  
Have a pre-recorded backup. A failed live demo with a working backup loses some marks under Presentation but does not affect Technical Depth or Production Quality if the code works when reviewed.

**Can a team of 2 submit a capstone?**  
Only with prior instructor approval and adjusted scope expectations. Teams of 5+ are not permitted.

**Do we need to deploy to the cloud?**  
No. Local deployment via Nexus is sufficient. Cloud deployment is welcome but not required or rewarded with extra marks.

**Can we use frameworks outside Kailash?**  
You must use at least 3 Kailash packages as specified. Additional non-Kailash tools (visualisation libraries, data sources, etc.) are fine as supplements, not replacements.
