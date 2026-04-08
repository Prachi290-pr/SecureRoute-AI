# SecureRoute-AI
## OpenEnv Environment for Enterprise Support Ticket Compliance

---

## Environment Description

SecureRoute-AI is a production-grade OpenEnv evaluation environment that simulates enterprise support ticket processing workflows. Agents receive raw customer support tickets containing mixed personally identifiable information (PII) and must perform two tasks:

1. **Redact** all sensitive data by replacing identified spans with `[REDACTED]`
2. **Route** each ticket to the correct department: `IT`, `BILLING`, or `SECURITY`

**Real-world context**: Enterprises process 10,000+ support tickets monthly containing credit card numbers, SSNs, API keys, emails, phone numbers, and other sensitive data. This environment evaluates agent capability to automatically sanitize tickets per compliance policy before human review or downstream LLM processing.

---

## Observation and Action Spaces

### Observation Space (Text Input)
Raw support ticket text as received by enterprise support teams:
Subject: Double charged invoice #INV-8472

Billing team,
I was charged $299 twice yesterday at 14:22 and 14:24. Transaction refs: TXN-XYZ123, TXN-ABC456.
Full details: Visa 4111-1111-1111-1111 exp 11/27
Cardholder: Michael Chen



### Action Space (Text Output + Routing Decision)
Structured response containing redacted ticket and department assignment:

redacted_ticket: "Subject: Double charged invoice #INV-8472\n\nBilling team,\n\nI was charged $299 twice yesterday at 14:22 and 14:24. Transaction refs: TXN-XYZ123, TXN-ABC456.\nFull details: Visa [REDACTED] exp 11/27\nCardholder: [REDACTED]"

department: "BILLING"


---

## Task Difficulty Levels

### Easy Tasks (5 tickets)
**Characteristics**: Single PII type or none, unambiguous routing signals  
**Examples**: 
- Password reset requests containing email addresses → `IT`
- Application crashes with no sensitive data → `IT`
**Agent requirements**: Basic pattern matching and keyword-based routing

### Medium Tasks (5 tickets)
**Characteristics**: 1-3 mixed PII types, financial context requiring precise redaction  
**Examples**:
- Double charges containing credit card numbers → `BILLING`
- Failed renewals with billing addresses → `BILLING`
**Agent requirements**: Multi-pattern PII detection, financial intent recognition

### Hard Tasks (5 tickets)
**Characteristics**: 3-5 complex PII types, security context, edge cases requiring context awareness  
**Examples**:
- Account compromise with SSNs, IP addresses, passwords → `SECURITY`
- Exposed API keys in communication channels → `SECURITY`
**Agent requirements**: Advanced security pattern recognition, nuanced context evaluation

---

## Evaluation Metrics

**Final episode score (0.0-1.0)**: `0.5 × Redaction Accuracy + 0.5 × Routing Accuracy`

### Redaction Accuracy Components
- **+1.0**: All PII correctly redacted, no over-redaction of harmless identifiers (transaction IDs, serial numbers)
- **+0.5**: Partial redaction (missed 1-2 spans)
- **0.0**: Any PII leakage detected

### Routing Accuracy Components
- **+1.0**: Correct department assignment (`IT`, `BILLING`, `SECURITY`)
- **0.0**: Incorrect department or no routing decision

**Reward density**: Intermediate rewards provided after each step() call, not only at episode termination.

---

## Implementation Details

**Dataset**: `tickets.json` contains 15 production-grade support tickets reflecting realistic PII distribution observed in enterprise environments.

**Grading**: 100% deterministic evaluation using regex pattern matching against ground truth PII spans and exact department matching.

**Episodes**: Multi-step interaction with maximum step budget of 12 steps per ticket.

**OpenEnv compliance**: Full implementation of `reset()`, `step(action)`, and `state()` interfaces with Pydantic-typed observation, action, and reward models.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/SecureRoute-AI.git
cd SecureRoute-AI

# Validate OpenEnv specification compliance
openenv validate .

# Run baseline model evaluation
export HF_TOKEN=your_token
python inference.py

# Test Docker containerization
docker build -t secureroute-ai .
docker run -p 7860:7860 secureroute-ai
```

---

## Deployment Configuration

**Hugging Face Spaces (Docker)**:
sdk: docker
app_port: 7860



**Container Requirements**:
- Python 3.9+
- Passes `openenv validate`
- Exposes evaluation endpoint on port 7860
- Reads `HF_TOKEN`, `MODEL_NAME`, `API_BASE_URL` from environment variables

---

## Project Structure

```plaintext
SecureRoute-AI/
├── tickets.json         # 15 raw production tickets
├── openenv.yaml         # OpenEnv metadata
├── models.py            # Pydantic observation/action/reward models
├── environment.py       # Core OpenEnv implementation
├── graders.py           # Deterministic scoring logic
├── inference.py         # Baseline evaluation script
├── Dockerfile           # HF Spaces deployment
└── README.md            # This document
```


---

## Environment Tags

`openenv`, `compliance`, `pii`, `enterprise`, `security`, `production`, `text`

---

**SecureRoute-AI evaluates production-critical agent capabilities for enterprise compliance workflows.**
