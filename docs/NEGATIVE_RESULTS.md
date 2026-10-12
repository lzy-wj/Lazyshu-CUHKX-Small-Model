# Negative Results and Lessons

Negative results are retained because they explain the final design better than the winning components alone.

## Larger trainable backbones

A modality-specialized visual backbone exposed roughly 117.2–178.4 million trainable high-level parameters. It rapidly forgot the cross-user representation. The project therefore moved to frozen backbones with zero-initialized residual adapters.

**Lesson:** more adaptation capacity can amplify target-domain overfitting.

## Official-style IMU Transformer

The reproduced IMU CNN-Transformer fitted training close to 99% but achieved only about 22.85%/24.68% on held-out subjects.

**Lesson:** optimization capacity was not the limiting factor; sensor placement and person shift dominated.

## Cross-fitted learned round Transformer

V109 reduced the V104 domain-protected anchor from 1175 to 1144 correct clips.

**Lesson:** learning a sequence correction across folds did not automatically transfer the right acquisition structure.

## Direct radar-time grouping

V160 improved local development from 1205 to 1222 correct clips but scored only 0.84079 publicly.

**Lesson:** timestamp grouping can encode a collection script rather than an activity mechanism.

## Repeated-transition consensus

V166 reached the project-high 1226/1357 and introduced no errors over 60 partial-script stress views, yet public accuracy fell to 0.88557.

**Lesson:** even strong cross-fold and stress-test evidence can remain conditional on the development data-generation process.

## Native high-resolution and repeated forward passes

Native 192/224-pixel candidates and diverse multi-forward inference did not provide a reliable gain over the 160-pixel protected system.

**Lesson:** extra compute without independent evidence is not useful under a fixed storage budget.

## Final design principle

New components were eventually treated as challengers rather than authorities. They could change a prediction only when supervision availability, label coverage, and independent neural evidence supported the change.
