# User Speculations & Project Hypotheses

This document tracks early speculations, risks, and hypotheses about the system's architecture and performance. These will be reviewed against actual system behavior once production is complete.

## 1. LLM Router Reliability (Added: Phase 0)
**Speculation:** Due to the LLM router that we are going to implement as a tie-breaker, it might fail to categorize the question correctly. If the LLM misclassifies or hallucinates the routing decision, that failure could cascade and break the entire application for that query.

*Review Checklist for Post-Production:*
- [ ] How often did the LLM router misclassify ambiguous questions?
- [ ] Did a misclassification cause an unhandled crash, or was it caught gracefully by the parameter gates?
- [ ] What percentage of queries relied on the LLM tie-breaker vs. the deterministic rules?

## 2. BigEarthNet Imagery Extraction Predicament (Added: Phase 0)
**Observation/Predicament:** While staging the data for the 200-step timing test, we discovered that the Hugging Face `BigEarthNet.txt` dataset is annotations-only and does not host the actual image pixels (to save bandwidth). The true imagery is locked inside a massive, monolithic 600GB+ LMDB archive on the official servers, which cannot be streamed selectively.
**Decision Made:** We intentionally used dummy (black) images for the 200-step test because compute timing relies only on tensor shapes, not pixel values. 
**Future Risk:** When we move to full training, we cannot use dummy data. We will be forced to either download the entire 600GB archive to extract our 80,000 manifest patches, or write a custom satellite API script to fetch the tiles manually via lat/long coordinates. 

*Review Checklist for Post-Production:*
- [ ] How did we ultimately extract the real images? (Full archive download vs. Sentinel API)
- [ ] Did the extraction bottleneck our training start time?
