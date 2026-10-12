# V91 Error Attribution

V89's remaining raw errors concentrate in fine-grained hand-object actions:
turning pages, writing, mobile-phone use, tableware use, taking medicine, and
reading documents. Higher resolution helps related classes such as phone calls,
drinking, eating, pouring, and peeling, but does not consistently solve the
whole group.

Test clips are nearly modality-complete: 394/405 contain all six modalities and
10 more miss only Thermal. Radar had already received zero transferred blend
weight in the historical blind-fold audit, so missing-modality handling and a
new Radar branch are not prioritized.

The next experiment therefore targets pseudo-label confirmation bias using
cross-scale agreement, rather than training another weak object specialist.
