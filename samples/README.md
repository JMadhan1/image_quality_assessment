# Sample images

Representative examples of each quality condition the system detects,
drawn from the generated synthetic dataset (ground truth known).

| File | Condition (ground truth) | Actual model output |
|---|---|---|
| 01_clean.jpg | clean / undistorted | 99.7, **ACCEPTABLE**, no issues |
| 02_blur.jpg | Gaussian blur, severity 5 | 30.2, **DEFECTIVE**, `blur` |
| 03_noise.jpg | Gaussian noise, severity 4 | 44.1, **DEGRADED**, `noise` |
| 04_overexposure.jpg | brightness +140, severity 5 | 20.7, **DEFECTIVE**, `overexposure` |
| 05_underexposure.jpg | brightness -55, severity 2 (mild) | 84.1, **ACCEPTABLE**, `blur` (misclassified) |
| 06_compression_corruption.jpg | heavy JPEG compression, severity 4 | 46.1, **DEGRADED**, `corruption` |
| 07_defect_blockcorrupt.jpg | random block corruption, severity 4 | 39.5, **DEFECTIVE**, `corruption` + `defect` |

Numbers above are actual verified outputs from the trained model (not
predicted/expected values) — reproduce with:
```bash
curl -X POST http://localhost:8000/analyze -F "file=@samples/02_blur.jpg"
```

Note on 05: this is a genuine, representative failure case, not cherry-picked
around it — it's the same weakness documented in
[EVALUATION.md](../EVALUATION.md#failure-case-analysis): a *mild* (severity 2)
brightness shift is subtle enough that the CNN misreads it as slight blur
instead. The overall score (84.1, ACCEPTABLE) is still reasonable since the
distortion genuinely is mild, but the specific issue type is wrong. Kept in
the sample set deliberately as an honest illustration of that limitation.

Upload any of these through the frontend, or via curl — see the "API
examples" section of the main [README](../README.md).
