# Resultats sur le jeu de test

| Modele | Parametres | Test loss | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CNN from scratch (SGD) | 1,173,730 | 0.6744 | 0.5684 | 0.6489 | 0.5684 | 0.5010 |
| Transfer learning (ResNet18) | 11,177,538 | 0.0688 | 0.9708 | 0.9710 | 0.9708 | 0.9708 |

- seed : 42
- device : cpu
- taille image : 224x224
- epochs : scratch 2, transfert 1 (+ 1 fine-tuning)
