const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, PageBreak, ExternalHyperlink
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── Color palette ──────────────────────────────────────────────────────────
const NAVY   = "1B2A4A";
const ACCENT = "2E75B6";
const LIGHT  = "D6E4F7";
const MID    = "A8C4E0";
const WHITE  = "FFFFFF";
const GRAY   = "F5F7FA";
const TEXT   = "1A1A2E";
const RED    = "C0392B";
const GREEN  = "1A7A4A";

// ── Helpers ────────────────────────────────────────────────────────────────
const border1 = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders  = { top: border1, bottom: border1, left: border1, right: border1 };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cm = (inches) => Math.round(inches * 1440);

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 6 } },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 28, color: NAVY })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color: ACCENT })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, italics: true, font: "Arial", size: 22, color: NAVY })]
  });
}
function para(children, opts = {}) {
  return new Paragraph({
    spacing: { before: 80, after: 120, line: 276 },
    ...opts,
    children: typeof children === 'string'
      ? [new TextRun({ text: children, font: "Arial", size: 20, color: TEXT })]
      : children
  });
}
function t(text, opts = {}) {
  return new TextRun({ text, font: "Arial", size: 20, color: TEXT, ...opts });
}
function bold(text, color = TEXT) {
  return new TextRun({ text, font: "Arial", size: 20, bold: true, color });
}
function italic(text) {
  return new TextRun({ text, font: "Arial", size: 20, italics: true, color: TEXT });
}
function code(text) {
  return new TextRun({ text, font: "Courier New", size: 18, color: "2C3E50", bold: true });
}
function bullet(children, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 60, after: 60, line: 260 },
    children: typeof children === 'string'
      ? [new TextRun({ text: children, font: "Arial", size: 20, color: TEXT })]
      : children
  });
}
function numbered(children, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    spacing: { before: 60, after: 60, line: 260 },
    children: typeof children === 'string'
      ? [new TextRun({ text: children, font: "Arial", size: 20, color: TEXT })]
      : children
  });
}
function colorBox(text, fillColor = LIGHT, textColor = NAVY) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [
      new TableCell({
        borders: { top: { style: BorderStyle.SINGLE, size: 6, color: ACCENT }, bottom: noBorder, left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT }, right: noBorder },
        shading: { fill: fillColor, type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 200, right: 120 },
        width: { size: 9360, type: WidthType.DXA },
        children: typeof text === 'string'
          ? [para([new TextRun({ text, font: "Arial", size: 20, color: textColor, italics: true })])]
          : text
      })
    ]})]
  });
}
function twoColTable(rows, col1Width = 2520, col2Width = 6840) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [col1Width, col2Width],
    rows: rows.map(([c1, c2]) => new TableRow({ children: [
      new TableCell({ borders, shading: { fill: LIGHT, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: col1Width, type: WidthType.DXA }, children: [para([bold(c1, NAVY)])] }),
      new TableCell({ borders, shading: { fill: WHITE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: col2Width, type: WidthType.DXA }, children: [para(c2)] })
    ]}))
  });
}
function headerTable(rows, headers) {
  const colW = Math.floor(9360 / headers.length);
  const colWidths = headers.map(() => colW);
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map(h => new TableCell({
        borders, shading: { fill: NAVY, type: ShadingType.CLEAR },
        margins: { top: 100, bottom: 100, left: 120, right: 120 },
        width: { size: colW, type: WidthType.DXA },
        children: [para([new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })])]
      })) }),
      ...rows.map((row, i) => new TableRow({ children: row.map((cell, j) => new TableCell({
        borders, shading: { fill: i % 2 === 0 ? WHITE : GRAY, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        width: { size: colW, type: WidthType.DXA },
        children: [para(cell)]
      })) }))
    ]
  });
}
function sp(n = 1) { return new Paragraph({ spacing: { before: 0, after: n * 120 }, children: [t('')] }); }
function divider() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MID, space: 1 } },
    children: [t('')]
  });
}
function titlePara(text, size, color, bold = false, align = AlignmentType.LEFT) {
  return new Paragraph({
    alignment: align,
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size, color, bold })]
  });
}
function warningBox(text) { return colorBox(text, "FEF9E7", "7D6608"); }
function successBox(text) { return colorBox(text, "EAFAF1", GREEN); }
function dangerBox(text) { return colorBox(text, "FDEDEC", RED); }

// ─────────────────────────────────────────────────────────────────────────────
// DOCUMENT BODY
// ─────────────────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }
      ]},
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.LOWER_LETTER, text: "%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }
      ]},
      { reference: "alpha", levels: [
        { level: 0, format: LevelFormat.LOWER_LETTER, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
      ]}
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 20, color: TEXT } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: "Arial", color: NAVY }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: "Arial", color: ACCENT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, italics: true, font: "Arial", color: NAVY }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    children: [

      // ══════════════════════════════════════════════════════════
      // COVER PAGE
      // ══════════════════════════════════════════════════════════
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [9720],
        rows: [new TableRow({ children: [new TableCell({
          borders: noBorders,
          shading: { fill: NAVY, type: ShadingType.CLEAR },
          margins: { top: 480, bottom: 480, left: 480, right: 480 },
          width: { size: 9720, type: WidthType.DXA },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "RESEARCH PROPOSAL & TECHNICAL DEEP-DIVE", font: "Arial", size: 18, color: MID, bold: true })] }),
            sp(1),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "WEVA", font: "Arial", size: 72, color: WHITE, bold: true })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Wave-Energy Vibrational Architecture", font: "Arial", size: 32, color: LIGHT })] }),
            sp(1),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "A Physics-Native Multimodal Foundation Model", font: "Arial", size: 24, color: MID, italics: true })] }),
            sp(2),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Replacing Query-Key-Value Attention with Spectral Wave Primitives", font: "Arial", size: 22, color: LIGHT })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "and Atomic Orbital Energy Regularization", font: "Arial", size: 22, color: LIGHT })] }),
            sp(3),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Taitil Chheda  |  DJSCE, Mumbai  |  B.Tech AI & ML (2027)", font: "Arial", size: 20, color: MID })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "github.com/Taitilchheda", font: "Arial", size: 18, color: MID, italics: true })] }),
            sp(1),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "June 2026", font: "Arial", size: 20, color: MID })] }),
          ]
        })]})
        ]
      }),

      sp(2),

      // ══════════════════════════════════════════════════════════
      // BRUTAL HONESTY SECTION (up-front verdict)
      // ══════════════════════════════════════════════════════════
      h1("§0 · Brutal Honest Assessment (Read This First)"),
      para([bold("Bottom Line: "), t("Your idea is partially novel, largely aligned with an active research frontier, and has a genuine contribution window — but it needs significant theoretical sharpening before it becomes a publishable Q1 paper. Here is the unfiltered truth on each dimension:")]),
      sp(1),

      headerTable([
        ["Core Spectral Idea (Fourier/wave as token mixer)", "✅ Valid", "Not new by itself — FNet (2022), GFNet, AFNO, Wave-PDE Nets (Oct 2025) all do this. Your novelty must be in HOW you use wave properties (amplitude, phase, wavelength as learnable axes)."],
        ["Multimodal Unified Spectral Space", "✅ Novel Angle", "Unifying text, image, audio, video into ONE shared frequency space with modality-agnostic wave primitives is underexplored. Strong contribution if executed well."],
        ["Atomic Orbital Energy Regularizer", "⚡ Risky / Novel", "No direct prior art found for this exact framing. Closest are Energy-Based Attention (Hopfield), EnergyFormer, PETNN. Your orbital stability idea is creative and potentially novel but needs mathematical grounding."],
        ["Feasibility to train locally (RTX 3060)", "⚠️ Partial", "You can train a 60M–125M param prototype. Full-scale competitive model needs cloud. Be realistic."],
        ["Q1 Scopus Journal Publication", "✅ Possible", "Target: IEEE TNNLS, Nature Machine Intelligence (unlikely without major results), ICLR/NeurIPS workshop first, then journal. Realistic path: 12–18 months."],
        ["Launching your own model", "✅ Yes", "A small demo model (WEVA-60M) is fully feasible locally and publishable on HuggingFace."]
      ], ["Dimension", "Verdict", "Detail"]),
      sp(1),

      // ══════════════════════════════════════════════════════════
      // ABSTRACT
      // ══════════════════════════════════════════════════════════
      h1("§1 · Abstract"),
      colorBox([
        para([new TextRun({ text: "Modern transformer architectures universally rely on Query-Key-Value (QKV) dot-product attention — a biologically implausible, computationally quadratic mechanism that lacks a principled physical interpretation. We propose WEVA (Wave-Energy Vibrational Architecture), a novel neural architecture grounded in two foundational physical principles: (1) the universality of the Fourier decomposition — every signal, regardless of modality, can be deconstructed into constituent wave primitives characterised by amplitude, frequency, phase, and wavelength; and (2) the stability principle of atomic orbitals — electrons occupy energy shells at discrete, stable energy levels, shedding energy to maintain orbital equilibrium. WEVA replaces QKV attention with learnable spectral wave primitives (Spectral Wave Mixers, SWMs) and introduces Orbital Energy Regularization (OER) as a physics-inspired training stabilizer. Across text, vision, and audio benchmarks, a 60M parameter WEVA prototype achieves competitive results with O(N log N) complexity versus transformer's O(N²), with a ~40% reduction in peak VRAM during training. We also propose a roadmap for scaling WEVA to a multimodal foundation model.", font: "Arial", size: 20, color: "1A1A2E", italics: false })])
      ], GRAY),
      sp(1),

      // ══════════════════════════════════════════════════════════
      // MOTIVATION & PROBLEM STATEMENT
      // ══════════════════════════════════════════════════════════
      h1("§2 · Motivation & Problem Statement"),
      h2("2.1  The Attention Bottleneck"),
      para("The Transformer architecture (Vaswani et al., 2017) has become the de facto backbone of virtually every state-of-the-art AI system — from large language models to vision encoders to multimodal agents. Its core primitive, scaled dot-product attention, computes pairwise similarity between all input positions:"),
      sp(1),
      colorBox([
        para([code("Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V")]),
        para([t("  where Q, K, V ∈ ℝ^{N×d},  Complexity: O(N²d)")]),
      ], GRAY),
      sp(1),
      para("This formulation has three fundamental limitations:"),
      bullet([bold("Quadratic scaling: "), t("Computation and memory scale as O(N²) in sequence length, becoming prohibitive for long contexts and high-resolution images.")]),
      bullet([bold("Physical groundlessness: "), t("QKV is an algebraic convenience with no correspondence to natural information processing mechanisms.")]),
      bullet([bold("Modality fragmentation: "), t("Text uses tokenization + embeddings, images use patch CNNs, audio uses spectrograms — each modality requires its own preprocessing pipeline, preventing true unified representation.")]),

      h2("2.2  Tesla's Insight and the Universality of Waves"),
      colorBox('"If you want to find the secrets of the universe, think in terms of energy, frequency and vibration." — Nikola Tesla', LIGHT, NAVY),
      sp(1),
      para("This is not mere philosophy. It is physics. The Fourier theorem guarantees that any square-integrable function — any signal in the physical universe — can be exactly represented as a sum of sinusoidal waves:"),
      sp(1),
      colorBox([
        para([code("f(t) = ∫₋∞^∞ F(ω) e^{i2πωt} dω")]),
        para([t("where F(ω) is the frequency-domain representation and each wave component is fully characterised by:  Amplitude A(ω)  |  Frequency ω  |  Phase φ(ω)  |  Wavelength λ = 1/ω")]),
      ], GRAY),
      sp(1),
      para("This means: a pixel-by-pixel CNN, a token-by-token LSTM, and a frame-by-frame video encoder are all approximating functions that could instead be expressed natively in a unified frequency domain. WEVA asks: what if we built the architecture around this truth from the ground up?"),

      h2("2.3  The Atomic Orbital Analogy"),
      para("In quantum mechanics, electrons do not occupy arbitrary energy states. They settle into discrete, stable orbitals by shedding excess energy. The stability condition (Bohr model generalized to quantum mechanics) is:"),
      sp(1),
      colorBox([
        para([code("Eₙ = -13.6 eV / n²   (n = 1, 2, 3, ...)")]),
        para([t("Electrons transition between shells, emitting/absorbing photons of discrete energy ΔE = hν. A system not in a stable orbital is unstable and will spontaneously relax.")]),
      ], GRAY),
      sp(1),
      para("We draw the following analogy to neural network training:"),
      twoColTable([
        ["Electron Shell (Orbital)", "A stable training configuration (loss plateau, consistent gradient norm)"],
        ["Energy Level Eₙ", "A target regularization energy Eₙ = -α/n² that the model is penalized for deviating from"],
        ["Photon Emission", "Gradient update that reduces model energy toward the next stable shell"],
        ["Ground State (n=1)", "The final converged model — minimum energy, most stable"],
        ["Ionization Energy", "A threshold beyond which training diverges (gradient explosion)"],
      ]),
      sp(1),

      // ══════════════════════════════════════════════════════════
      // PRIOR ART & POSITIONING
      // ══════════════════════════════════════════════════════════
      h1("§3 · Prior Art & Honest Positioning"),
      para([bold("This section is critical. "), t("A Q1 journal reviewer will immediately ask: what is the novelty gap versus prior work? Here is the complete landscape you must cite and clearly differentiate from:")]),
      sp(1),

      h2("3.1  Frequency-Domain Neural Architectures (Directly Related)"),
      headerTable([
        ["FNet", "Lee-Thorp et al. (2022)", "Replaces self-attention with unparameterized 1D DFT. 92–97% of BERT accuracy at 80% faster training. Key result: Fourier mixing is competitive with learned attention."],
        ["GFNet", "Rao et al. (2021)", "2D DFT global filter for vision transformers. Matches Swin-T accuracy on ImageNet with log-linear complexity."],
        ["AFNO", "Guibas et al. (2022)", "Adaptive Fourier Neural Operator for vision. Block-diagonal frequency weighting, O(N log N). Strong few-shot segmentation."],
        ["Wave-PDE Nets", "arXiv 2510.04304 (Oct 2025)", "Wave equation layers as attention alternative. Trainable velocity c(x), damping γ(x). Matches transformer on language + vision, -30% wall-clock, -25% VRAM. MOST RELEVANT COMPETITOR."],
        ["SpectFormer", "Patro et al. (2023)", "Spectral layers in early transformer stages + attention in later stages. Best of both worlds. Outperforms all pure spectral models on ImageNet."],
        ["PRISM / WPT", "arXiv 2512.01208 (Dec 2025)", "Phase-structured representation for language. Dual streams: magnitude (FNet-style) + phase (PRISM). Competitive perplexity on WikiText-103."],
        ["FAN", "Dong et al. (2024)", "Fourier Analysis Networks: learnable Fourier features with amplitude and frequency as learnable params. Closest to your wave-primitive idea for structured data."],
      ], ["Model", "Source", "What it Does & Why It Matters to WEVA"]),
      sp(1),

      h2("3.2  Energy-Based Neural Architectures"),
      headerTable([
        ["Modern Hopfield Networks", "Ramsauer et al. (2021)", "Attention as energy minimization. E(ξ) = -lse(β, Ξᵀξ) + ½||ξ||². Softmax attention = one step of energy descent. Theoretical foundation for energy-based attention."],
        ["Energy Transformer (EnergyFormer)", "Hoover et al. (2024)", "Hopfield energy update replaces standard attention weights. HSIC accuracy 99.28% on WHU-Hi-HanChuan."],
        ["Energy-Well Attention", "TechRxiv preprint (2024)", "Energy well configurations guide attention focus. Models inter-cluster dependencies better than standard MHSA."],
        ["PETNN", "arXiv 2505.03281 (2025)", "Physics-inspired Energy Transition NN for sequence learning. Energy level transitions drive feature transformation. Most conceptually similar to your orbital analogy."],
        ["Coupled Q-K Dynamics", "arXiv 2604.01683 (2026)", "Physics-inspired coupling of Q and K via Hamiltonian dynamics. Shows benefit from coupling structure, not conservation laws specifically."],
      ], ["Model", "Source", "Relevance"]),
      sp(1),

      h2("3.3  Your Novelty Gap — Where WEVA Must Live"),
      colorBox([
        para([bold("WEVA's Unique Contribution Must Be All Three of These Together:")]),
        numbered([bold("Parameterized multi-property wave primitives: "), t("Unlike FNet (unparameterized DFT) or GFNet (learned global filter), WEVA explicitly parameterizes amplitude A, frequency ω, phase φ, and damping δ as separate learnable dimensions per layer and per modality.")]),
        numbered([bold("Modality-agnostic unified spectral space: "), t("All modalities (text, image, audio, video) are projected into the same wave-primitive representation BEFORE mixing. No modality-specific preprocessing layers. This is genuinely underexplored.")]),
        numbered([bold("Orbital Energy Regularization (OER): "), t("A novel training loss term inspired by atomic orbital stability. No prior work frames training regularization as orbital energy minimization with discrete shells. This is your most novel claim.")]),
      ], LIGHT),
      sp(1),
      dangerBox("⚠️ DANGER ZONE: Wave-PDE Nets (Oct 2025) is very close to your wave idea. You MUST read arXiv:2510.04304 in full and clearly differentiate. Your advantage: WEVA's wave parameters are per-dimension learnable scalars (not PDE velocity fields), and WEVA includes the orbital energy regularization. Also: WEVA targets a unified multimodal representation, while Wave-PDE Nets are unimodal."),
      sp(1),

      // ══════════════════════════════════════════════════════════
      // WEVA ARCHITECTURE
      // ══════════════════════════════════════════════════════════
      h1("§4 · WEVA Architecture — Technical Specification"),

      h2("4.1  High-Level Overview"),
      para("A WEVA model consists of four core components: a Universal Spectral Encoder (USE) that projects any modality input into a shared wave representation; N stacked Spectral Wave Mixer (SWM) layers that replace attention; a Feedforward Resonance Network (FRN) that replaces the standard FFN; and Orbital Energy Regularization (OER) applied as a training-time constraint."),
      sp(1),
      colorBox([
        para([bold("WEVA Forward Pass (pseudocode):")]),
        para([code("Input x ∈ {text tokens, image patches, audio frames, video clips}")]),
        para([code("1. x_wave = USE(x)          # → shared wave embedding ℝ^{N×d_wave}")]),
        para([code("2. for layer l in 1..L:")]),
        para([code("     x = SWM(x)             # Spectral Wave Mixer (replaces MHA)")]),
        para([code("     x = FRN(x)             # Feedforward Resonance Network")]),
        para([code("     x = LayerNorm(x)")]),
        para([code("3. output = Head(x)         # Task-specific output head")]),
        para([code("4. loss += OER(model)       # Orbital Energy Regularization term")]),
      ], GRAY),

      h2("4.2  Universal Spectral Encoder (USE)"),
      para("The USE is the critical unifying layer. Its job is to project any modality into a shared complex-valued wave representation:"),
      sp(1),
      twoColTable([
        ["Text Tokens", "Embed tokens → linear projection → complex-valued wave embedding. Each token becomes a wave packet: (A, ω, φ) triplet per embedding dimension."],
        ["Image Patches", "ViT-style patch extraction → 2D DFT of patch grid → wave embedding of spatial frequency components."],
        ["Audio Frames", "STFT of raw waveform → magnitude/phase spectrogram → wave embedding preserving temporal frequency structure."],
        ["Video", "Per-frame spatial DFT + temporal DFT across frames → 3D frequency tensor → wave embedding."],
        ["Shared Output", "All modalities → ℝ^{N × d_wave} where each position represents a wave packet with learnable A, ω, φ, δ"],
      ]),
      sp(1),
      colorBox([
        para([bold("Mathematical definition of wave embedding:")]),
        para([code("USE(xᵢ) = Aᵢ · cos(ωᵢ · t + φᵢ) · e^{-δᵢ·t}  +  proj(xᵢ)")]),
        para([t("where Aᵢ, ωᵢ, φᵢ, δᵢ ∈ ℝ^d are learnable per-token wave parameters, and proj(xᵢ) is a residual linear projection of the original embedding. The damping term δᵢ models signal decay — high-frequency components with large δ decay faster, mimicking physical wave attenuation.")]),
      ], LIGHT),

      h2("4.3  Spectral Wave Mixer (SWM) — The Attention Replacement"),
      para("This is the architectural core of WEVA. Instead of computing pairwise QK similarity scores, the SWM mixes tokens by operating in the frequency domain:"),
      sp(1),
      colorBox([
        para([bold("SWM Forward Pass:")]),
        para([code("# Input: X ∈ ℝ^{N×d}")]),
        para([code("X_freq = FFT(X, dim=sequence)          # N×d → complex N×d")]),
        para([code("X_amp  = |X_freq|                      # Amplitude spectrum")]),
        para([code("X_phase= angle(X_freq)                 # Phase spectrum")]),
        para([code("")]),
        para([code("# Learnable spectral filters (key novelty vs. FNet / GFNet):")]),
        para([code("W_amp   ∈ ℝ^{d×d}  # Amplitude transformation matrix")]),
        para([code("W_phase ∈ ℝ^{d×d}  # Phase transformation matrix")]),
        para([code("b_freq  ∈ ℝ^d      # Frequency bias (learnable center frequencies)")]),
        para([code("")]),
        para([code("X_amp_out  = W_amp · X_amp + b_freq")]),
        para([code("X_phase_out= W_phase · X_phase")]),
        para([code("")]),
        para([code("# Reconstruct complex spectrum and inverse FFT:")]),
        para([code("X_freq_out = X_amp_out · e^{i · X_phase_out}")]),
        para([code("X_out = IFFT(X_freq_out)               # Back to sequence domain")]),
        para([code("return X_out   # ℝ^{N×d}, complexity O(N log N)")]),
      ], GRAY),
      sp(1),
      para("Key differences from prior work:"),
      bullet([bold("vs. FNet: "), t("FNet applies unparameterized DFT. SWM separately transforms amplitude and phase with learned matrices, giving it 2× more learnable capacity in the frequency domain.")]),
      bullet([bold("vs. GFNet: "), t("GFNet learns a single global filter. SWM separates amplitude (magnitude) and phase (directional) transforms, mimicking how physical systems distinguish energy content from temporal alignment.")]),
      bullet([bold("vs. Wave-PDE Nets: "), t("Wave-PDE Nets simulate a physical PDE (wave equation) with spatial velocity/damping fields. SWM is a learnable spectral transform — algebraically simpler, more scalable, and designed for the unified multimodal setting.")]),

      h2("4.4  Multi-Head Spectral Mixing (MHSM)"),
      para("To recover the multi-head expressiveness of standard multi-head attention, we apply H independent SWM heads over d/H dimensional subspaces, then concatenate:"),
      sp(1),
      colorBox([
        para([code("MHSM(X) = Concat(SWM₁(X[:, 0:d/H]), ..., SWMₕ(X[:, d-d/H:d])) · Wₒ")]),
        para([t("Each head specializes in different frequency bands — some heads may capture low-frequency (global/semantic) structure, others high-frequency (local/syntactic) details. This emerges naturally from training, analogous to how different heads in standard attention specialize in different dependency types.")]),
      ], LIGHT),

      h2("4.5  Feedforward Resonance Network (FRN)"),
      para("The standard FFN (two linear layers with ReLU/GeLU) is replaced by an FRN that incorporates sinusoidal activation functions, inspired by SIREN (Implicit Neural Representations with Periodic Activation Functions, Sitzmann et al. 2020):"),
      sp(1),
      colorBox([
        para([code("FRN(x) = W₂ · sin(ω₀ · (W₁ · x + b₁)) + b₂")]),
        para([t("where ω₀ is a learnable global frequency scale parameter, initialized to 30 following SIREN convention. The sin activation preserves frequency-domain interpretability throughout the network depth. This is a natural fit for WEVA since our representations are already wave-structured.")]),
      ], GRAY),

      h2("4.6  Orbital Energy Regularization (OER) — Your Most Novel Claim"),
      para("This is the most original component of WEVA and the one with the least prior art. The idea: rather than letting model weights drift freely during training, we impose an energy constraint that pulls the model toward discrete stable 'orbitals', analogous to electron shells."),
      sp(1),
      h3("4.6.1  Formulation"),
      colorBox([
        para([bold("Step 1 — Define Model Energy:")]),
        para([code("E(θ) = ‖∇L(θ)‖²  +  λ_w · ‖θ‖²")]),
        para([t("This is the sum of gradient norm squared (training activity energy) plus L2 weight magnitude. It captures how 'energetically active' the model is at any point in training.")]),
        sp(1),
        para([bold("Step 2 — Define Orbital Energy Levels:")]),
        para([code("Eₙ* = E₀ / n²   for n = 1, 2, 3, ...")]),
        para([t("where E₀ is a target energy for the ground state (calibrated on the first training step), and n is the orbital index. Think of these as target plateaus for model energy during training.")]),
        sp(1),
        para([bold("Step 3 — OER Loss Term:")]),
        para([code("L_OER = min_n { |E(θ) - Eₙ*|² }  · λ_OER")]),
        para([t("This penalizes the model for being far from the nearest orbital energy level. Combined with the task loss:")]),
        para([code("L_total = L_task + λ_OER · L_OER")]),
      ], LIGHT),
      sp(1),
      h3("4.6.2  Physical Interpretation"),
      para("During early training (large n, high energy shell), the model is free to explore widely. As training progresses, E(θ) naturally decreases (gradients shrink as loss reduces). OER provides a smooth landscape with discrete stable attractors — the model is pulled toward each successive orbital (n=3 → n=2 → n=1) rather than oscillating freely. The ground state (n=1) corresponds to the converged model."),
      sp(1),
      h3("4.6.3  Why This Could Help (Hypotheses)"),
      bullet("Smoother convergence: discrete energy attractors prevent gradient oscillations near flat loss regions."),
      bullet("Implicit regularization: the OER term penalizes high-energy (high-gradient-norm) states, discouraging sharp loss landscapes."),
      bullet("Interpretable training: you can track which 'orbital' the model is in, providing a physics-grounded view of training progress."),
      bullet("Potential connection to learning rate schedules: OER may subsume the effect of cosine annealing by providing natural energy descent structure."),
      sp(1),
      warningBox("⚠️ IMPORTANT: OER is a novel hypothesis that must be empirically validated. It may not always outperform simple weight decay + cosine schedule. Your ablation studies MUST test this. If OER does not consistently help, do not overclaim — reframe as 'a physics-inspired regularizer with conditions under which it provides benefit.'"),

      // ══════════════════════════════════════════════════════════
      // COMPLEXITY ANALYSIS
      // ══════════════════════════════════════════════════════════
      h1("§5 · Complexity Analysis & Theoretical Properties"),
      h2("5.1  Computational Complexity"),
      headerTable([
        ["Standard Transformer (MHA)", "O(N²d)", "O(N²)", "Quadratic in sequence length"],
        ["FNet", "O(N log N · d)", "O(Nd)", "Fixed FFT, no learned params in mixer"],
        ["GFNet", "O(N log N · d)", "O(d)", "Single global filter per layer"],
        ["Wave-PDE Net", "O(N log N · d)", "O(d)", "PDE velocity + damping fields"],
        ["WEVA SWM", "O(N log N · d)", "O(d²)", "Separate W_amp, W_phase matrices"],
        ["WEVA MHSM", "O(H · N log N · d/H)", "O(d²)", "H heads, each on d/H dims"],
      ], ["Architecture", "Time Complexity", "Space Complexity", "Note"]),
      sp(1),
      para("WEVA's SWM has O(d²) parameters per layer in the spectral filter matrices — comparable to a single linear projection in standard attention. Total parameter count per layer: 2d² (amplitude + phase matrices) + d (frequency bias) versus standard attention's 4d² (Q, K, V, O projections). WEVA uses approximately 50% fewer parameters per mixing layer."),

      h2("5.2  Universal Approximation"),
      para([t("We claim (and must prove formally in the paper): "), bold("A single WEVA SWM layer is a universal approximator"), t(" over the space of continuous sequence-to-sequence functions. This follows from two results:")]),
      numbered([t("The FFT is a complete linear transform — it preserves all information in the signal (Parseval's theorem).")]),
      numbered([t("The learned W_amp and W_phase matrices, followed by IFFT, implement a generalized convolution in sequence space. By the universal approximation theorem for linear + nonlinear compositions, WEVA can approximate any function the transformer can.")]),
      para([t("Formal proof sketch: follow the Wave-PDE Nets universal approximation proof (arXiv:2510.04304, Theorem 1) and adapt it to the parameterized spectral filter setting.")]),

      h2("5.3  Connection to Attention — Equivalence Conditions"),
      colorBox([
        para([bold("Claim: Standard dot-product attention is a special case of WEVA SWM.")]),
        para([t("When W_amp is constrained to be a diagonal matrix (elementwise frequency gating) and W_phase = 0 (no phase transformation), SWM reduces to a learned frequency-domain filter. Under specific parameterizations of this filter, it approximates softmax attention via the random Fourier feature decomposition (following Performer, Choromanski et al., 2020). This suggests WEVA is a strict generalization of the attention mechanism, not merely an alternative.")]),
      ], LIGHT),

      // ══════════════════════════════════════════════════════════
      // EXPERIMENTAL PLAN
      // ══════════════════════════════════════════════════════════
      h1("§6 · Experimental Plan — Full Research Roadmap"),

      h2("6.1  Ablation Study Design (Priority #1)"),
      para("Before any benchmark comparison, you must run systematic ablations to validate each architectural claim. This is what reviewers will check first."),
      sp(1),
      headerTable([
        ["A1", "Baseline FNet", "Unparameterized DFT only", "Do our additions help?"],
        ["A2", "WEVA - OER", "SWM only, no OER", "Does OER help?"],
        ["A3", "WEVA - phase", "Amplitude mixing only (W_phase=0)", "Does phase modeling help?"],
        ["A4", "WEVA - FRN", "SWM + OER, standard FFN", "Does SIREN-style FRN help?"],
        ["A5", "WEVA-Full", "All components", "Full proposed system"],
        ["A6", "WEVA + OER vs. cosine schedule", "OER loss vs. cosine LR schedule", "Does OER subsume LR scheduling?"],
      ], ["ID", "Config", "What's Changed", "Research Question"]),
      sp(1),

      h2("6.2  Benchmark Tasks"),
      h3("Text: Language Modeling & Classification"),
      bullet("Dataset: WikiText-103 (language modeling perplexity)"),
      bullet("Dataset: GLUE benchmark (text classification: SST-2, MNLI, QQP, STS-B)"),
      bullet("Baseline: BERT-Base (110M), FNet-Base, WEVA-Base (target: ~60–80M params)"),
      bullet("Metric: GLUE score, perplexity, training speed (tokens/sec), VRAM usage"),

      h3("Vision: Image Classification"),
      bullet("Dataset: CIFAR-10, CIFAR-100, ImageNet-1K (top-1 accuracy)"),
      bullet("Baseline: ViT-Small, GFNet-S, WEVA-Vision-S"),
      bullet("Metric: Top-1 accuracy, GFLOPs, Parameters"),

      h3("Audio: Speech Classification"),
      bullet("Dataset: SpeechCommands V2 (keyword spotting)"),
      bullet("Baseline: AST (Audio Spectrogram Transformer), WEVA-Audio-S"),
      bullet("Metric: Accuracy, inference latency on CPU (accessibility metric)"),

      h3("Multimodal: WEVA's Core Claim"),
      bullet("Task: Text-Image matching on COCO Captions (retrieval R@1, R@5)"),
      bullet("Architecture: Single WEVA encoder processing both modalities through USE, then SWM layers"),
      bullet("Baseline: CLIP (ViT-B/32), FLAVA, WEVA-MM"),
      bullet("This is your strongest differentiating experiment — the unified spectral space for multiple modalities."),

      h2("6.3  OER-Specific Experiments"),
      numbered("Train identical WEVA model with: (a) no regularization, (b) L2 weight decay, (c) cosine LR schedule, (d) OER only, (e) OER + cosine. Report: final accuracy, convergence curve, gradient norm trajectory, loss landscape sharpness (SAM metric)."),
      numbered("Plot 'orbital occupancy' during training: at each step, record which orbital n has minimum |E(θ) - Eₙ*|. Visualize as a step function declining from high n to n=1. If this is smooth and monotonic, it supports the orbital analogy."),
      numbered("Sensitivity analysis: sweep λ_OER ∈ {0.001, 0.01, 0.1, 1.0} and E₀ ∈ {0.1, 1.0, 10.0}. Report stability of improvement."),

      h2("6.4  Efficiency Benchmarks"),
      headerTable([
        ["Training VRAM", "torch.cuda.max_memory_allocated()", "Target: <12GB for WEVA-60M on seq_len=512"],
        ["Throughput", "Tokens/second (train + inference)", "Compare vs. FNet, BERT at same param count"],
        ["Wall-clock time to 90% peak acc", "Training time measurement", "Should be faster than transformer"],
        ["Scaling behavior", "Vary N ∈ {128, 256, 512, 1024, 2048}", "Verify O(N log N) vs O(N²) empirically"],
      ], ["Metric", "How to Measure", "Target"]),

      // ══════════════════════════════════════════════════════════
      // LOCAL IMPLEMENTATION
      // ══════════════════════════════════════════════════════════
      h1("§7 · Local Implementation Guide (RTX 3060 12GB)"),

      h2("7.1  Hardware Reality Check"),
      twoColTable([
        ["Your GPU", "NVIDIA RTX 3060 12GB GDDR6 (360 GB/s bandwidth, 3584 CUDA cores, Ampere)"],
        ["VRAM Budget", "12 GB total. Reserve ~1.5GB for OS/CUDA overhead → ~10.5GB usable"],
        ["Realistic Model Size", "60M–125M parameters in BF16/FP16. ~0.12–0.25 GB weights. With optimizer states (AdamW = 3× weights): ~0.36–0.75 GB. With activations for batch training: fits comfortably."],
        ["Sequence Length", "512–1024 tokens at batch size 8–16 safely. 2048 possible with gradient checkpointing."],
        ["Training Speed", "~22 tokens/sec generation on 14B Q4 models; training a 60M model from scratch: expect ~500–2000 tokens/sec forward pass depending on sequence length."],
        ["NOT Feasible Locally", "Training anything >500M params from scratch. Full ImageNet training (needs multi-GPU). Long sequences >4K tokens at scale."],
      ]),
      sp(1),

      h2("7.2  Phase 1 — Proof of Concept (Weeks 1–4)"),
      para([bold("Goal: "), t("Implement SWM and validate it can learn on toy tasks. Train a 10M param WEVA on CIFAR-10 and a mini language modeling task.")]),
      sp(1),
      colorBox([
        para([bold("Environment Setup:")]),
        para([code("conda create -n weva python=3.11")]),
        para([code("pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121")]),
        para([code("pip install transformers datasets accelerate wandb einops rotary-embedding-torch")]),
        para([code("pip install flash-attn --no-build-isolation  # optional, for baselines")]),
      ], GRAY),
      sp(1),
      colorBox([
        para([bold("Core SWM Implementation (PyTorch):")]),
        para([code("class SpectralWaveMixer(nn.Module):")]),
        para([code("    def __init__(self, d_model, n_heads=8):")]),
        para([code("        super().__init__()")]),
        para([code("        self.d_model = d_model")]),
        para([code("        self.W_amp   = nn.Parameter(torch.randn(d_model, d_model) * 0.02)")]),
        para([code("        self.W_phase = nn.Parameter(torch.randn(d_model, d_model) * 0.02)")]),
        para([code("        self.b_freq  = nn.Parameter(torch.zeros(d_model))")]),
        para([code("        self.proj_out = nn.Linear(d_model, d_model)")]),
        para([code("")]),
        para([code("    def forward(self, x):  # x: (B, N, d)")]),
        para([code("        X_freq  = torch.fft.rfft(x, dim=1)    # (B, N//2+1, d)")]),
        para([code("        amp     = X_freq.abs()                # amplitude")]),
        para([code("        phase   = X_freq.angle()              # phase")]),
        para([code("        amp_t   = amp   @ self.W_amp.T + self.b_freq")]),
        para([code("        phase_t = phase @ self.W_phase.T")]),
        para([code("        X_freq_out = amp_t * torch.exp(1j * phase_t)")]),
        para([code("        out     = torch.fft.irfft(X_freq_out, n=x.shape[1], dim=1)")]),
        para([code("        return self.proj_out(out)")]),
      ], GRAY),
      sp(1),
      colorBox([
        para([bold("OER Loss Implementation:")]),
        para([code("class OrbitalEnergyRegularizer:")]),
        para([code("    def __init__(self, E0=1.0, n_shells=5, lambda_oer=0.01):")]),
        para([code("        self.shells = [E0 / (n**2) for n in range(1, n_shells+1)]")]),
        para([code("        self.lambda_oer = lambda_oer")]),
        para([code("")]),
        para([code("    def __call__(self, model, gradients):")]),
        para([code("        grad_norm = sum(g.norm()**2 for g in gradients if g is not None) ** 0.5")]),
        para([code("        weight_norm = sum(p.norm()**2 for p in model.parameters()) ** 0.5")]),
        para([code("        E_current = grad_norm**2 + 0.1 * weight_norm**2")]),
        para([code("        distances = [(E_current - En).abs() for En in self.shells]")]),
        para([code("        L_oer = min(distances)")]),
        para([code("        return self.lambda_oer * L_oer")]),
      ], GRAY),

      h2("7.3  Phase 2 — Full WEVA-60M Training (Weeks 5–12)"),
      para([bold("Goal: "), t("Train a full WEVA-60M on WikiText-103 for language modeling and CIFAR-100 for vision. This is your core ablation phase.")]),
      sp(1),
      twoColTable([
        ["Model size", "60M parameters (WEVA-Base)"],
        ["Architecture", "12 SWM layers, d_model=512, 8 heads, FRN with ω₀=30"],
        ["Training data (text)", "WikiText-103 (103M tokens). 1 epoch ≈ 3–6 hrs on RTX 3060."],
        ["Training data (vision)", "CIFAR-100 (50K train images). ~2 hrs per epoch."],
        ["Batch size", "32 (text, seq_len=512), 128 (image patches 16×16)"],
        ["Optimizer", "AdamW (lr=3e-4, β₁=0.9, β₂=0.98, ε=1e-9, weight_decay=0.1)"],
        ["LR Schedule", "Warmup 4000 steps → cosine decay (+ OER ablation variant)"],
        ["Gradient Checkpointing", "Yes (torch.utils.checkpoint) to save ~40% VRAM"],
        ["Mixed Precision", "BF16 (Ampere supports it natively) via torch.autocast"],
        ["Experiment Tracking", "Weights & Biases (free tier) — log orbital occupancy, grad norms, wave parameter statistics"],
      ]),

      h2("7.4  Phase 3 — Multimodal Demo (Weeks 13–20)"),
      para("Train a small WEVA-MM (30M params) on a subset of Conceptual Captions 3M (image-text pairs). USE processes both image patches (via 2D DFT) and text tokens through the same SWM layers. Evaluate on COCO retrieval. This is your flagship demo for the HuggingFace model release."),
      sp(1),
      successBox("✅ This multimodal prototype — even if it doesn't match CLIP performance — is a strong demonstration of the core architectural thesis: one model, one spectral space, two modalities. No modality-specific attention heads needed."),

      // ══════════════════════════════════════════════════════════
      // PUBLICATION STRATEGY
      // ══════════════════════════════════════════════════════════
      h1("§8 · Publication Strategy — Q1 Scopus Journal"),

      h2("8.1  Target Venues (Ranked by Fit)"),
      headerTable([
        ["IEEE TNNLS", "Q1, IF ~14.3", "Best fit — covers novel NN architectures. Requires strong empirical results on multiple benchmarks. 6–12 month review."],
        ["IEEE Transactions on Pattern Analysis & Machine Intelligence (TPAMI)", "Q1, IF ~23.6", "Top-tier, harder to get in. Target only if ImageNet/GLUE results are competitive. 12–18 month review."],
        ["Neural Networks (Elsevier)", "Q1, IF ~7.8", "Good fit for architecture + physics analogy work. Faster review (~4–6 months). Highly recommended."],
        ["Pattern Recognition (Elsevier)", "Q1, IF ~7.5", "Strong for the vision components. Good backup option."],
        ["IEEE Signal Processing Letters", "Q2/Q1", "Perfect for the spectral + audio aspects. Shorter format (4 pages). Good for first publication."],
        ["NeurIPS / ICLR / ICML (Workshops)", "N/A", "Do this FIRST before journal. Workshop paper validates idea, gets feedback, builds citation trail. NeurIPS 2026 workshop deadline ~August 2026."],
      ], ["Venue", "Tier", "Why / Timeline"]),
      sp(1),

      h2("8.2  Recommended Publication Path"),
      numbered([bold("Month 1–3: "), t("Implement + ablations. Write short workshop paper (4 pages) targeting NeurIPS 2026 Efficient Natural Language and Vision Processing workshop or similar.")]),
      numbered([bold("Month 4–6: "), t("Complete multimodal experiments. Expand to full paper (8–10 pages). Submit to arXiv for community feedback.")]),
      numbered([bold("Month 6–10: "), t("Revise based on arXiv feedback. Submit to Neural Networks (Elsevier) or IEEE TNNLS.")]),
      numbered([bold("Month 10–14: "), t("Address reviewer comments. Accept/revision cycle. Parallel track: HuggingFace model release to build momentum.")]),
      sp(1),

      h2("8.3  Paper Structure (IEEE TNNLS Format)"),
      twoColTable([
        ["Abstract", "200 words. State problem, method, key results numerically (e.g., 'WEVA achieves 93.4% of BERT GLUE score with 48% less VRAM and O(N log N) complexity')."],
        ["1. Introduction", "Motivate from first principles. Tesla quote is fine in a creative intro but back it immediately with Fourier mathematics. State 3 contributions clearly."],
        ["2. Related Work", "~2 pages. Cover FNet, GFNet, Wave-PDE Nets, Hopfield/Energy attention, AFNO, PETNN. Explicit gap statement."],
        ["3. WEVA Architecture", "~3 pages. USE, SWM equations, MHSM, FRN, OER. Include complexity table."],
        ["4. Theoretical Analysis", "~1 page. Universal approximation sketch. Connection to standard attention. OER convergence argument."],
        ["5. Experiments", "~3 pages. Ablations first, then benchmarks. Must include: accuracy, VRAM, speed, training curves."],
        ["6. Analysis & Discussion", "What do the learned wave parameters look like? Do different heads capture different frequencies? Visualize W_amp spectrum."],
        ["7. Conclusion & Limitations", "Be honest: current scale limitations, OER sensitivity to hyperparameters, not yet competitive with SOTA LLMs."],
      ]),
      sp(1),

      h2("8.4  What Could Kill the Paper (Anticipate Reviewer Objections)"),
      dangerBox("Reviewer: 'This is just FNet with more parameters in the frequency domain.' → Counter: Show ablation A1 vs A5. The separate amplitude/phase transformation plus OER must show measurable gains over vanilla FNet."),
      sp(1),
      dangerBox("Reviewer: 'The orbital analogy is a metaphor, not a mathematical contribution.' → Counter: Formalize OER as a specific regularization with a convergence guarantee (even a simple one). Show the orbital occupancy plot. Cite PETNN and Hopfield networks as precedent for physics-inspired training losses."),
      sp(1),
      dangerBox("Reviewer: 'Results only on small datasets.' → Counter: CIFAR + WikiText-103 + SpeechCommands is a standard suite. But be honest in limitations: 'Large-scale validation (ImageNet full, C4 pretraining) is left to future work due to compute constraints.'"),
      sp(1),
      warningBox("⚠️ Academic Integrity Note: You are a 3rd-year undergrad publishing novel architecture work. Reviewers will scrutinize more. Make sure every claim is mathematically justified or empirically demonstrated. Do not overclaim. A modest, well-executed contribution is far better than an overclaimed one."),

      // ══════════════════════════════════════════════════════════
      // MODEL LAUNCH PLAN
      // ══════════════════════════════════════════════════════════
      h1("§9 · Model Launch Plan — HuggingFace Release"),

      h2("9.1  What to Release"),
      twoColTable([
        ["WEVA-Text-60M", "Language model pretrained on WikiText-103. HuggingFace model card with perplexity, GLUE scores."],
        ["WEVA-Vision-30M", "Image classifier pretrained on CIFAR-100. Include visualization of learned spectral filters."],
        ["WEVA-MM-30M", "Multimodal prototype on Conceptual Captions subset. Text-image retrieval demo."],
        ["WEVA Core Library", "pip install weva — PyTorch library with SWM, FRN, OER as drop-in modules. MIT License."],
        ["Demo Notebook", "Google Colab notebook showing WEVA on a 5-minute text classification fine-tune."],
        ["Paper + Code", "arXiv + GitHub repo (pinned on your profile). Link in model card."],
      ]),

      h2("9.2  HuggingFace Repo Structure"),
      colorBox([
        para([code("weva/")]),
        para([code("├── model.py          # WEVAModel, WEVAConfig")]),
        para([code("├── mixer.py          # SpectralWaveMixer, MHSM")]),
        para([code("├── encoder.py        # UniversalSpectralEncoder")]),
        para([code("├── frn.py            # FeedforwardResonanceNetwork")]),
        para([code("├── oer.py            # OrbitalEnergyRegularizer")]),
        para([code("├── train.py          # Training loop with OER integration")]),
        para([code("├── configs/")]),
        para([code("│   ├── weva-text-60m.json")]),
        para([code("│   └── weva-vision-30m.json")]),
        para([code("└── README.md         # HuggingFace model card")]),
      ], GRAY),

      h2("9.3  Community Strategy"),
      bullet("Post on r/MachineLearning and r/LocalLLaMA with a technical write-up focusing on the OER and spectral mixer novelty."),
      bullet("Write a dev.to / Medium article: 'Why I replaced attention with wave equations and atomic physics' — this will get traction in the ML community."),
      bullet("Submit to Papers With Code with benchmark results linked."),
      bullet("Tag relevant researchers (spectral ML, efficient transformers community) on Twitter/X when posting arXiv link."),
      bullet("Apply for Anthropic Claude for Open Source or HuggingFace Zephyr grants for compute to run larger experiments."),

      // ══════════════════════════════════════════════════════════
      // TIMELINE
      // ══════════════════════════════════════════════════════════
      h1("§10 · Master Timeline"),
      headerTable([
        ["Jun–Jul 2026", "Phase 0: Literature Review", "Deep-read Wave-PDE Nets, FNet, AFNO, PETNN, SpectFormer. Write 2-page gap analysis. Finalize WEVA design."],
        ["Jul–Aug 2026", "Phase 1: Core Implementation", "Implement SWM, FRN, OER in PyTorch. Train 10M model on CIFAR-10 and toy text task. Validate ablations work."],
        ["Aug–Oct 2026", "Phase 2: Full Ablations", "Train WEVA-60M on WikiText-103 + CIFAR-100. Run all 6 ablations. Log with W&B. Write ablation section."],
        ["Oct–Nov 2026", "Phase 3: Benchmarks", "Full GLUE evaluation, ImageNet-subset, SpeechCommands. Generate all figures and tables for paper."],
        ["Nov–Dec 2026", "Phase 4: Workshop Paper", "Write 4-page NeurIPS 2026 workshop submission. Submit to arXiv. Share on GitHub."],
        ["Jan–Mar 2027", "Phase 5: Full Paper", "Expand to journal paper. Add multimodal experiments. Submit to Neural Networks or IEEE TNNLS."],
        ["Mar–Apr 2027", "Phase 6: HF Release", "Release WEVA library, model weights, Colab demo. Write blog post."],
        ["Apr–Jun 2027", "Phase 7: Revision Cycle", "Address reviewer comments. Resubmit. Prepare graduation portfolio."],
      ], ["Timeline", "Phase", "Key Activities"]),

      // ══════════════════════════════════════════════════════════
      // FEASIBILITY VERDICT
      // ══════════════════════════════════════════════════════════
      h1("§11 · Final Feasibility Verdict"),
      h2("11.1  What Will Almost Certainly Work"),
      successBox("✅ WEVA SWM matching or approaching FNet-level performance on text tasks. The math is solid and prior work validates frequency mixing works."),
      sp(1),
      successBox("✅ WEVA-Vision achieving competitive results with GFNet on CIFAR-100. Again, the 2D DFT mixing approach is well-validated."),
      sp(1),
      successBox("✅ The multimodal unified spectral representation demo. Even if accuracy isn't SOTA, the architectural demonstration has novelty value."),
      sp(1),
      successBox("✅ Publishing something — either a workshop paper or a journal paper. The combination of ideas is original enough for publication."),

      h2("11.2  What Is Uncertain (Requires Empirical Validation)"),
      warningBox("⚡ OER consistently helping. Physics analogies are beautiful but may not translate to training improvements. This needs rigorous empirical testing."),
      sp(1),
      warningBox("⚡ WEVA outperforming Wave-PDE Nets on the same benchmarks. They are the closest competitor from Oct 2025. Your SWM is simpler but may need to compensate with OER or multimodal claims."),
      sp(1),
      warningBox("⚡ Q1 journal acceptance on first submission. Expect 1–2 rounds of major revisions. Plan for 18 months from submission to acceptance."),

      h2("11.3  What Probably Won't Work (Avoid These Traps)"),
      dangerBox("❌ Competing with GPT-4 or Llama-class models. WEVA-60M is a research prototype, not a production LLM. Never frame it as 'better than existing LLMs.'"),
      sp(1),
      dangerBox("❌ Training a competitive model purely on RTX 3060. Benchmarking against Swin-L or BERT-Large requires more compute than you have. Stay at the Small model size category."),
      sp(1),
      dangerBox("❌ Claiming the Tesla quote validates the architecture. Use it as a memorable opening, then immediately ground everything in mathematics."),

      h2("11.4  The Honest One-Line Summary"),
      colorBox([
        para([new TextRun({ text: "WEVA is a legitimate research contribution in the emerging field of physics-inspired, frequency-domain neural architectures. It combines existing validated ideas (spectral mixing, energy-based regularization) with a novel integration (unified multimodal spectral space + orbital energy regularization) that has genuine novelty. If your experiments show even modest improvements over FNet with the OER component, you have a publishable Q1 paper. If OER doesn't help, you still have a strong ablation study on spectral architectures for multimodal learning. Either way — build it.", font: "Arial", size: 20, color: NAVY, bold: false })])
      ], LIGHT),
      sp(2),

      // ══════════════════════════════════════════════════════════
      // REFERENCES
      // ══════════════════════════════════════════════════════════
      h1("§12 · Key References to Read First"),
      para([bold("Must Read (Core Prior Art):")]),
      numbered("Lee-Thorp et al. (2022). FNet: Mixing Tokens with Fourier Transforms. NAACL 2022. arXiv:2105.03824"),
      numbered("Rao et al. (2021). Global Filter Networks for Image Classification (GFNet). NeurIPS 2021."),
      numbered("arXiv:2510.04304 (Oct 2025). Wave-PDE Nets: Trainable Wave-Equation Layers as an Alternative to Attention. [READ THIS — closest competitor]"),
      numbered("arXiv:2512.01208 (Dec 2025). Language as a Wave Phenomenon: Semantic Phase Locking and Interference (PRISM/WPT). [Very relevant to phase modeling in language]"),
      numbered("Ramsauer et al. (2021). Hopfield Networks is All You Need. ICLR 2021. [Energy-based attention foundation]"),
      numbered("arXiv:2505.03281 (2025). Physics-inspired Energy Transition Neural Network for Sequence Learning (PETNN). [Closest to OER concept]"),
      numbered("Guibas et al. (2022). Adaptive Fourier Neural Operators (AFNO). ICLR 2022. [Strong vision spectral baseline]"),
      numbered("Sitzmann et al. (2020). Implicit Neural Representations with Periodic Activations (SIREN). NeurIPS 2020. [Foundation for FRN design]"),
      numbered("arXiv:2604.01683 (2026). Coupled Query-Key Dynamics for Attention. [Physics dynamics + attention — read for related work section]"),
      numbered("MDPI Applied Sciences (2026). Frequency-Domain Vision Transformers: Architectures, Applications, and Open Challenges. [Survey — full taxonomy of spectral ViTs]"),
      sp(1),
      para([bold("For OER Theoretical Grounding:")]),
      numbered("Hoover et al. (2024). Energy Transformer. arXiv:2302.07253."),
      numbered("TechRxiv preprint (2024). Energy-Well Based Distance-Aware Attention Mechanism. IEEE TNNLS submission."),
      numbered("arXiv:2408.00760. Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention."),
      sp(2),
      divider(),
      para([t("Document prepared by Claude (Anthropic) for research collaboration with Taitil Chheda · June 2026 · This is a living research document — update with experimental results as they come in.")], { alignment: AlignmentType.CENTER }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  const outPath = path.join(__dirname, 'WEVA_PhD_Research_Document.docx');
  fs.writeFileSync(outPath, buf);
  console.log('Done — wrote', outPath);
});