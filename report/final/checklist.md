
\clearpage

# NeurIPS Paper Checklist {-}

The questions below are the NeurIPS 2025 paper checklist, verbatim; it serves as this paper's
reproducibility checklist. Section numbers refer to this paper.

```{=latex}
\begin{enumerate}

\item {\bf Claims}
    \item[] Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
    \item[] Answer: \answerYes{}
    \item[] Justification: Each claim in the abstract maps to a result: transfer failure to Section~\ref{sec:results} (Tables~\ref{tbl:main}--\ref{tbl:verdict}); the learned cue to Section~\ref{sec:why} (Tables~\ref{tbl:taxonomy}--\ref{tbl:saliency}); robustness, false-positive and cost figures to Section~\ref{sec:deploy}. The scope (detection on one held-out surface, frozen detectors, no end-to-end attack success) is stated in Sections~\ref{sec:intro} and~\ref{sec:conclusion}.

\item {\bf Limitations}
    \item[] Question: Does the paper discuss the limitations of the work performed by the authors?
    \item[] Answer: \answerYes{}
    \item[] Justification: Section~\ref{sec:conclusion} (Limitations) covers the small holdouts, the narrow B2 search and seed variance, single-seed analyses, the AI-validated codebook, baseline dependence of integrated gradients, and the absence of an end-to-end evaluation.

\item {\bf Theory assumptions and proofs}
    \item[] Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
    \item[] Answer: \answerNA{}
    \item[] Justification: The paper has no theoretical results.

\item {\bf Experimental result reproducibility}
    \item[] Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
    \item[] Answer: \answerYes{}
    \item[] Justification: Sections~\ref{sec:data} and~\ref{sec:method} give the sources, group splits, features, hyperparameters, search spaces, thresholds and interval methods; Appendix~\ref{app:reproduce} lists the commands in order. Seeds are fixed, and committed per-row score files reproduce every AUROC and deployed FPR in Table~\ref{tbl:main}.

\item {\bf Open access to data and code}
    \item[] Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
    \item[] Answer: \answerYes{}
    \item[] Justification: The code, configuration, the B2 training notebook, the result tables and per-row scores of every detector are in the public repository (\url{https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense}), with instructions in its README and Appendix~\ref{app:reproduce}. Access is partial in one respect: the rebuilt dataset, the B2 Colab score files and the B2 checkpoint sit in a team-only DVC remote, because some upstream sources forbid redistribution. Outside readers rebuild the dataset with the DVC pipeline from upstream repositories pinned to exact commits (NewsQA must be built under its own terms) and retrain B2 with the notebook (Section~\ref{sec:data}, Appendix~\ref{app:reproduce}).

\item {\bf Experimental setting/details}
    \item[] Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
    \item[] Answer: \answerYes{}
    \item[] Justification: Section~\ref{sec:data} (splits, Table~\ref{tbl:data}) and Section~\ref{sec:method} (features, search spaces and objectives, selected values, DeBERTa learning rate, batch size, epochs, selection rule, threshold rule). Full search settings are in the repository (\texttt{report/phase4/b1\_optuna/best\_params.json}).

\item {\bf Experiment statistical significance}
    \item[] Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
    \item[] Answer: \answerYes{}
    \item[] Justification: AUROC carries 95\% percentile-bootstrap intervals that resample whole groups (so BIPIA twin pairs never split); FPR carries Wilson 95\% intervals; B2 is reported as mean $\pm$ standard deviation over three seeds, with per-seed intervals for the pre-registered criterion (Section~\ref{sec:method}, Tables~\ref{tbl:main}, \ref{tbl:verdict}, \ref{tbl:fairness}). Small groups (MSB, CJK rows) are flagged as indicative.

\item {\bf Experiments compute resources}
    \item[] Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
    \item[] Answer: \answerYes{}
    \item[] Justification: Table~\ref{tbl:cost} and Section~\ref{sec:deploy} give inference time per row on an Apple M4 Max laptop (CPU and GPU), fit times, the 55-minute Optuna search on one CPU core, and the B2 runs on a Colab T4. B2's training time was not recorded; the paper gives the run plan's estimate (6--9 GPU-hours) and labels it as one.

\item {\bf Code of ethics}
    \item[] Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics \url{https://neurips.cc/public/EthicsGuidelines}?
    \item[] Answer: \answerYes{}
    \item[] Justification: The work uses only public benchmarks, involves no human subjects, respects source licenses by not redistributing data, and reports harms and mitigations (Sections~\ref{sec:data} and~\ref{sec:deploy}).

\item {\bf Broader impacts}
    \item[] Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
    \item[] Answer: \answerYes{}
    \item[] Justification: Section~\ref{sec:deploy} (Table~\ref{tbl:harms}) lists six harms (false sense of security, blocking legitimate tool authors, dual use, the dataset as an attack corpus, over-reliance, opaque blocking), each with evidence and a mitigation. The positive impact, a cheap, documented first-pass defense and an honest account of where it fails, is the subject of Section~\ref{sec:conclusion} and Appendix~\ref{app:card}.

\item {\bf Safeguards}
    \item[] Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
    \item[] Answer: \answerYes{}
    \item[] Justification: The rebuilt dataset, which collects working injection strings, stays in a private remote; the repository commits scores and row ids rather than text; red-team results are released as aggregates. Only the TF-IDF detectors can be refit from the public code alone; the B2 checkpoint stays team-only. The recommended detector has a model card that states its out-of-scope uses (Section~\ref{sec:deploy}, Appendix~\ref{app:card}).

\item {\bf Licenses for existing assets}
    \item[] Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
    \item[] Answer: \answerYes{}
    \item[] Justification: Every benchmark and model is cited (Sections~\ref{sec:data}--\ref{sec:method}). PoisonedRAG, BIPIA and MSB are MIT-licensed, BIPIA's tables CC BY-SA 4.0, and NewsQA and XSum carry research-use terms; MCPTox states no license, so it is used internally only and not redistributed. Licenses are listed per source in the repository's data card (\texttt{DATA\_CARD.md}).

\item {\bf New assets}
    \item[] Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
    \item[] Answer: \answerYes{}
    \item[] Justification: The new assets are the code, the dataset build pipeline and the trained detectors. They are documented by the repository README, a data card in the style of datasheets and Data Cards, and a model card (Appendix~\ref{app:card}).

\item {\bf Crowdsourcing and research with human subjects}
    \item[] Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
    \item[] Answer: \answerNA{}
    \item[] Justification: No crowdsourcing or human subjects; every row comes from public benchmarks.

\item {\bf Institutional review board (IRB) approvals or equivalent for research with human subjects}
    \item[] Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
    \item[] Answer: \answerNA{}
    \item[] Justification: There are no study participants. Section~\ref{sec:data} explains why the work is not human-subjects research under 45 CFR 46.102.

\item {\bf Declaration of LLM usage}
    \item[] Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.
    \item[] Answer: \answerYes{}
    \item[] Justification: Language models are components of two detectors (DistilGPT2 in B0, DeBERTa-v3 in B2; Section~\ref{sec:method}). An AI assistant wrote parts of the code and the error codebook and made the codebook's 40-row agreement check (Section~\ref{sec:taxonomy}); this is declared in Section~\ref{sec:conclusion}.

\end{enumerate}
```
