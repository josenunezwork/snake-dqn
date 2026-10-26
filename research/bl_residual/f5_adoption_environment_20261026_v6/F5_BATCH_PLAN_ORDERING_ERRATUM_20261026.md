# F5 batch-plan ordering clarification

Prospective create-only clarification of the common-environment amendment SHA-256 `20df1c5ec2a80d2f81ce4711c8bbb163814c181a0fe3de29573042465ef28064` and machine environment SHA-256 `bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5`.

The original F5 v2 R3 ordering remains binding: after the legitimate mechanism selection, materialize and seal the common 512,000-row batch plan BEFORE ANY COV COMPLETION OR FITTING. Both arms bind the same ordered row identities and initial/final generator states. COV completion may fill only the selected original action labels and cannot regenerate, reorder, re-stratify or change plan membership. Exact implementation/materialization and timing qualification remain later execution gates.

This clarifies the amendment's abbreviated wording 'before either fit'; it changes no scientific rule and authorizes no diagnostic, completion, fit or execution. Original bytes remain unchanged.
