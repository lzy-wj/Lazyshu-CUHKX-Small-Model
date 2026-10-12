# V165 Balanced-Cycle Radar

V165 extracts only class-count-preserving label permutations from the Radar
round decoder. Within each round it accepts the largest changed subset whose
V127 and V160 label multisets are identical. This repairs temporal position
swaps without changing the predicted action inventory. The final variant takes
the union with V164's independent clip-evidence gate.
