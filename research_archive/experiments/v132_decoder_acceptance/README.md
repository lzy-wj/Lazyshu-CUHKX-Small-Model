# V132 Decoder-Acceptance Gate

V127's sequence constraints provide most of its gain, but 12 remaining errors
have the correct class as the routed raw top-1 prediction. V132 learns no model
and changes no sequence template. It selects one simple decoder-acceptance rule
on Fold 0: retain a sequence rewrite only when the routed clip probabilities
provide sufficient rank or log-margin support, or revert high-confidence raw
predictions. The unchanged V127 control is eligible and the selected rule is
transferred once to Fold 1.

