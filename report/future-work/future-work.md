# Future work notes

Paragraphs for the final report's Future Work section. They are written in the same
pandoc markdown as `report/phase2/report.md`. To use one, paste it into the report and
add its BibTeX entry, listed below, to the report's bibliography.

## Skill and agent registries as a fourth surface

Our surfaces are RAG passages, tool outputs and MCP tool descriptions. Coding agents
now also load *skills* and *agent definitions*: instruction files that the agent reads
and follows, and that teams share through registries. The open ai-registry
specification [@atlan2026airegistry] is one example. It pins each skill version,
verifies that every machine installs identical bytes, and records approvals and
rollbacks. These controls show *which* text was installed. They do not show whether
that text carries an injection, so a poisoned skill that passes review reaches every
agent that installs it. Like an MCP tool description, a skill file is trusted
instruction text that the user never wrote. Two follow-ups would test this:

- evaluate the detector on skill files as an additional LOSO surface, which first needs
  a labeled set of benign and poisoned skills;
- run the detector inside the registry's publish and approval step, so a skill is
  scanned before it is approved rather than after it is installed.

### BibTeX

```bibtex
@misc{atlan2026airegistry,
  title = {ai-registry: An Open Specification for {AI} Skill and Agent Registries},
  author = {{Atlan}},
  year = {2026},
  howpublished = {\url{https://github.com/atlanai/ai-registry}},
  note = {Specification draft v0.1. Accessed 2026-11-09},
}
```
