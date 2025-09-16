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

## Dlaczego wybrane metryki

- **Precision, Recall, F1**: dobrze oddają jakość detekcji przy nierównych klasach; F1 daje jeden wskaźnik do szybkiego porównania.
- **TP/FP/FN**: surowe zliczenia ułatwiają debugowanie (czego brakuje vs. co jest nadmiarowe).
- **mean IoU**: mierzy dokładność lokalizacji sparowanych ramek; uzupełnia F1 (które nie mówi, jak dobrze pasują ramki).
- **AP@0.5**: podsumowuje zależność precision–recall w funkcji progu pewności przy standardowym IoU=0.5; szybkie i powszechne wstępne kryterium (zamiast cięższego COCO mAP@[.5:.95]).
- **IoU=0.5 (domyślnie)**: rozsądne dla dokumentów, ale można zacieśnić (np. 0.6–0.75) parametrem `iou_threshold`.
- **Dopasowanie etykiet**: trafienie liczone tylko przy zgodnym typie (np. Text vs Table), żeby ocena odzwierciedlała semantykę, nie tylko geometrię.

