# 🧬 Advanced Machine Learning — Project 5 Personalization Guide
> **Semantic Correspondence with Visual Foundation Models (DINOv2 + LoRA)**

Personalizzare uno scheletro di progetto è il modo migliore per farlo risaltare e dimostrare una comprensione profonda della materia. Basandomi sui requisiti del **Project 5 (Semantic Correspondence)** e sulle tendenze attuali del Machine Learning, ecco i requisiti principali e diverse alternative di personalizzazione per ciascuno.

---

## 🏛️ 1. Stage 1: Analisi del Backbone (Baseline)
**Il Requisito:** Confrontare come diversi modelli (DINOv2, SAM, DINOv3) estraggono feature per la corrispondenza senza addestramento.

*   **Alt A: Analisi "Layer-wise" Profonda.** Non limitarti a usare l'ultimo layer. Confronta le performance estraendo feature dai layer iniziali, intermedi e finali. Visualizza come la "semantica" emerge man mano che si sale nei blocchi del Transformer.
*   **Alt B: Confronto Q-K-V.** Invece di usare i semplici patch tokens, confronta l'uso di **Keys**, **Queries** o **Values** delle teste di attenzione. Spesso le *Keys* degli ultimi layer sono le più discriminative per il matching.
*   **Alt C: Domain-Specific Benchmark.** Invece di fare una media generica su SPair-71k, crea dei sotto-benchmark: quanto bene performa DINOv2 sulle categorie "Animali" (organico) rispetto a "Veicoli" (rigido/geometrico)? Documenta quali modelli falliscono su quali domini.
*   **Alt D: Visualizzazione delle Attention Maps.** Sovrapponi le mappe di attenzione di DINOv2 all'immagine originale (es. l'attenzione del token [CLS]). Questo aiuta a capire se il modello sta effettivamente "guardando" l'oggetto o lo sfondo durante il matching.
*   **Alt E: Analisi di Equivarianza Geometrica.** Ruota o scala l'immagine target e osserva come varia il PCK. Un buon backbone deve essere coraggiosamente equivariante rispetto a trasformazioni rigide.

---

## 🔧 2. Stage 2: Fine-Tuning Efficiente (LoRA/Adapters)
**Il Requisito:** Adattare il modello pre-addestrato al task di corrispondenza usando pochi parametri (LoRA).

*   **Alt A: Contrastive Correspondence Loss.** Invece della MSE (regressione), implementa una **InfoNCE Loss**. Tratta il punto target come un esempio positivo e tutti gli altri patch come negativi. È lo stato dell'arte per imparare rappresentazioni robuste.
*   **Alt B: Confronto tra PEFT (LoRA vs Adapters).** Implementa sia LoRA che degli "Adapter" classici (piccoli MLP dopo i blocchi di attenzione). Analizza quale dei due è più efficiente in termini di memoria e velocità di convergenza.
*   **Alt C: Curriculum Learning.** Inizia il fine-tuning su coppie di immagini "facili" (stessa posa, scala simile) e aumenta gradualmente la difficoltà (viewpoint diversi). Questo stabilizza molto l'addestramento di LoRA.
*   **Alt D: Loss Pesata per Difficoltà dei Keypoint.** Usa una funzione di costo che penalizzi maggiormente gli errori sui keypoint che il modello baseline non riusciva a individuare (es. parti piccole dell'oggetto).
*   **Alt E: Augmentation con TPS (Thin Plate Splines).** Genera nuove coppie di training applicando deformazioni elastiche alle immagini originali. Questo insegna al modello a essere robusto a deformazioni non rigide.

---

## 🎯 3. Stage 3: Raffinamento della Predizione (Sub-pixel)
**Il Requisito:** Sostituire l'argmax discreto con il Window Soft-Argmax per una precisione superiore.

*   **Alt A: Adaptive Windowing.** Invece di una finestra fissa (es. 5x5), rendi la dimensione della finestra dinamica basata sull'entropia della mappa di similarità: se il modello è "incerto" (mappa piatta), usa una finestra più grande.
*   **Alt B: Fitting Gaussiano 2D.** Invece del soft-argmax (media pesata), prova a fittare una superficie Gaussiana 2D sulla mappa di similarità attorno al picco per trovare il massimo teorico a livello sub-pixel.
*   **Alt C: Multi-Scale Pyramid Matching.** Calcola la similarità a diverse scale del backbone (es. layer 8 e layer 12). Usa la scala grossolana per trovare la regione e quella fine per il raffinamento sub-pixel.
*   **Alt D: Feature Smoothing Locale.** Applica un filtro (es. sfocatura gaussiana) alla mappa di similarità prima dell'argmax. Questo riduce l'effetto del rumore locale nei patch ad alta frequenza e stabilizza la predizione.
*   **Alt E: Iterative Confidence Refinement.** Esegui una prima predizione, calcola la similarità locale e, se la confidenza è bassa, sposta la finestra di ricerca e riesamina l'area vicina con una temperatura softmax più bassa.

---

## 🌟 4. Stage 4: Extension (Personalizzazione Libera)
**Il Requisito:** Un'estensione obbligatoria che esplori nuove direzioni.

*   **Alt A: "Zero-Shot" Cross-Domain Matching.** Valuta quanto il tuo modello addestrato su foto reali performa su disegni o sketch (usando ad esempio il dataset *CUB-200-Painting*). È un test di robustezza semantica molto apprezzato.
*   **Alt B: Segment-Aware Correspondence.** Integra **SAM (Segment Anything)**. Invece di cercare il match in tutta l'immagine, usa la maschera dell'oggetto in cui si trova il keypoint sorgente per filtrare i possibili match nell'immagine target. Riduce drasticamente i "falsi positivi".
*   **Alt C: Demo Interattiva con Gradio.** Crea un'interfaccia web (usa la libreria `gradio`) dove l'utente può caricare due immagini, cliccare su un punto e vedere istantaneamente il match predetto dal tuo modello con LoRA + Soft-Argmax. Questo dà al progetto un look molto professionale.
*   **Alt D: Video Object Tracking via Correspondence.** Applica il tuo modello a sequenze video (dataset DAVIS). Traccia i keypoint da un frame all'altro usando la corrispondenza semantica al posto dei classici tracker.
*   **Alt E: Cross-Category/Cross-Species Matching.** Testa se il modello è in grado di mappare i keypoint di un gatto su quelli di un cane, o di una giraffa su quelli di un cavallo. Analizza quanto è "generale" la semantica appresa.

---

## 🛠️ Stack Tecnologico Consigliato
*   **Training**: PyTorch + Lightning (opzionale) per la gestione dei checkpoint.
*   **Logging**: Weights & Biases per visualizzare i match durante il training.
*   **Sub-pixel**: `torch.meshgrid` per creare la finestra attorno al picco del soft-argmax.
*   **Backbones**: `torch.hub.load('facebookresearch/dinov2', ...)` o libreria `timm`.

---
> [!IMPORTANT]
> **Consiglio Strategico**: Per ottenere la lode, scegli un'alternativa per ogni stage. Ad esempio:
> **Layer-wise Analysis** (Stage 1) + **InfoNCE Loss** (Stage 2) + **Adaptive Windowing** (Stage 3) + **Interactive Demo** (Extension).
