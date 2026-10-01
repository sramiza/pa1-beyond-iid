# PA1: Beyond i.i.d. Learning

Programming Assignment 1 for Advanced Techniques in ML (EE-5102 / CS-6304) at LUMS. The four tasks look at different ways a model's test data can stop looking like its training data: which visual cues a model relies on, unsupervised domain adaptation, domain generalization, and open-set recognition. All experiments were run in Kaggle GPU notebooks.

Report: [`report/PA1_Report_Beyond_iid_tracked_final.pdf`](report/PA1_Report_Beyond_iid_tracked_final.pdf) (12 pages plus references).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── report/
│   └── PA1_Report_Beyond_iid_tracked_final.pdf
├── task1_inductive_biases/
│   └── task1_notebook.ipynb
├── task2_domain_adaptation/
│   ├── task2_notebook.ipynb                  # main run: CDAN, DAN, separability, per-class, λ_MMD sweep
│   └── task2_run1_erm_dann_training.ipynb    # earlier run: ERM and DANN training logs
├── task3_domain_generalization/
│   └── task3_notebook.ipynb
└── task4_open_set_recognition/
    └── task4_notebook.ipynb
```

## Running the notebooks

The notebooks were written for Kaggle and use its paths (`/kaggle/input`, `/kaggle/working`). Each one is saved with the outputs from the run used in the report.

- **Datasets.** STL-10 (Task 1) and CIFAR-10/100 (Task 4) download through torchvision. Tasks 2 and 3 use the Kaggle PACS dataset `nickfratto/pacs-dataset`.
- **Task 1** installs `open_clip_torch` and `scikit-image` itself. It caches the trained linear heads, the extracted features and the cue-conflict images in `/kaggle/working`, so a rerun skips any stage that is already done.
- **Task 2** has two notebooks from the same Kaggle session on 22 September. `task2_run1_erm_dann_training.ipynb` trained ERM and DANN (its CDAN run was cut off at epoch 16). `task2_notebook.ipynb` then loaded those ERM and DANN weights from `/kaggle/working` instead of retraining them, so its ERM and DANN training cells are commented out; it retrained CDAN and ran DAN, the separability probe, the per-class analysis and the λ_MMD sweep. To rerun Task 2 from scratch, uncomment those two training cells.
- **Task 3** loads the ERM checkpoint from Task 2 (`source_only_erm.pth`) from the Kaggle dataset set in `ERM_CHECKPOINT_PATH`; change that path if you upload the checkpoint somewhere else. Sketch is only loaded for the final evaluation, after every training and selection decision is made.
- **Task 4** skips training for any model whose checkpoint already exists, and resumes an interrupted run from its per-epoch resume file.

## Results at a glance

Short versions; the tables, figures and caveats are in the report.

**Task 1: Inductive biases (STL-10).** Frozen ResNet-50 and ViT-B/16 (both ImageNet-trained) and CLIP ViT-B/32 (OpenAI weights), each with a linear head, plus CLIP zero-shot. We test them under grayscale, hue rotation, Gatys-style shape/texture cue-conflict images (300 of them), translation and 4x4 patch shuffling, and compare representations with cosine stability (I_T) and t-SNE. ResNet-50's shape bias on the cue-conflict images is 53.2%, against 82.6% for ViT and 77.8% / 82.2% for CLIP (head / zero-shot). Under patch shuffling, ViT is the most robust (a 7.6-point drop) and CLIP's head the least (22.8 points, with ResNet at 10.8), even though both ViT and CLIP are ViT-B models.

**Task 2: Unsupervised domain adaptation (PACS, Sketch as the unlabeled target).** Sketch accuracy / macro-F1: ERM 72.46% / 74.41%, DAN 73.58% / 71.76%, DANN 30.54% / 14.34%, CDAN 35.71% / 37.05%. CDAN pushed its domain classifier down to chance and still did far worse than ERM, predicting "person" for most elephant and dog sketches. DANN's training diverged during its first epoch, even with a looser AdamW eps and gradient clipping, so its numbers come from that epoch-1 checkpoint. A separate logistic-regression probe could still tell source features from target features 92-100% of the time for every method.

**Task 3: Domain generalization (PACS, Sketch never seen).** ERM, DAN-DG (MMD alignment across the three source domains) and SAM. DAN-DG and SAM both beat ERM on source-domain validation but did worse on Sketch (71.01% and 59.38% accuracy, against 72.46%). SAM found the flattest minimum (Δsharp 0.2059, against 0.4146 for ERM) yet had the worst Sketch accuracy, with most of its mistakes going to "dog".

**Task 4: Open-set recognition (CIFAR-10 known, CIFAR-100 near/far unknowns).** A CIFAR-ResNet-18 trained as Vanilla, GCSC (RandAugment) and PROSER, scored with MSP, MLS, Energy and Mahalanobis. PROSER's placeholder score gives the best pooled AUROC (0.867), and Mahalanobis is the best post-hoc score (0.859). GCSC lost 7.6 points of closed-set accuracy and had the worst near and pooled AUROC. Every method/score combination is 0.08-0.15 AUROC worse on near unknowns than on far unknowns.

## Environment

Python 3.10+ and PyTorch with CUDA. `requirements.txt` lists the packages the notebooks import; versions weren't pinned, and the runs used whatever Kaggle's image provided at the time.

## Author

Syed Ramiz Abbas ([@sramiza](https://github.com/sramiza)), LUMS CS.
