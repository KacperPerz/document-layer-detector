## Build and run

```sh
make build
make run
make wait
```

## One-shot evaluate (prints metrics and writes compare.png)

```sh
make evaluate
```

Use your own files (overrides):

```sh
make evaluate EVAL_IMG=/absolute/path/to/your_image.png EVAL_ANN=/absolute/path/to/your_annotations.json IOU=0.5
```

