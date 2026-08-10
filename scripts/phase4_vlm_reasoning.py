#!/usr/bin/env python3
"""
Phase 4: VLM Reasoning Layer
LLaVA-1.5 7B (4-bit quantized) generates explainable verdicts
from detection + OCR + ASR results
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLMReasoningEngine:
    """LLaVA-based reasoning for violation verdicts"""

    def __init__(self):
        """Initialize LLaVA-1.5 7B with 4-bit quantization"""
        logger.info("=" * 70)
        logger.info("PHASE 4: VLM REASONING LAYER (LLaVA-1.5 7B)")
        logger.info("=" * 70)

        logger.info("\n1️⃣ Loading LLaVA-1.5 7B model...")
        logger.info("   Model: LLaVA-1.5-7B-hf")
        logger.info("   Quantization: 4-bit (bitsandbytes)")
        logger.info("   GPU Memory: ~4GB")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            logger.info("\n   Loading model (this may take 1-2 minutes)...")

            self.model_id = "llava-hf/llava-1.5-7b-hf"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logger.info("   ✓ Model loaded successfully")
            logger.info(f"   ✓ Quantization: 4-bit NF4")
            logger.info(f"   ✓ Device: {self.model.device}")

        except Exception as e:
            logger.error(f"   ✗ Failed to load LLaVA: {e}")
            logger.info("\n   💡 Fallback: Using rule-based reasoning")
            self.model = None
            self.tokenizer = None

        logger.info("\n✅ VLM Engine ready!\n")

    def generate_reasoning_prompt(self, detection: Dict, ocr_text: Optional[str] = None) -> str:
        """Generate structured prompt for VLM"""
        class_name = detection.get("class_name", "unknown")
        confidence = detection.get("confidence", 0.0)

        prompt = f"""Analyze this violation detection and provide structured reasoning.

DETECTION INFO:
- Class: {class_name}
- Confidence: {confidence:.1%}
- Bounding Box: {detection.get('bbox_xyxy', 'N/A')}
"""

        if ocr_text:
            prompt += f"\nEXTRACTED TEXT:\n- {ocr_text}\n"

        prompt += """
TASK: Provide a structured JSON verdict for content moderation.

Return ONLY valid JSON (no markdown, no extra text):
{
  "violation_type": "weapon|nsfw|counterfeit|safe",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 sentence explanation",
  "evidence": ["list", "of", "supporting", "factors"],
  "recommended_action": "flag|review|allow"
}
"""
        return prompt

    def rule_based_reasoning(self, detection: Dict, ocr_text: Optional[str] = None) -> Dict:
        """Fallback rule-based reasoning"""
        class_name = detection.get("class_name", "unknown").lower()
        confidence = detection.get("confidence", 0.0)

        # Rule-based logic
        if "weapon" in class_name:
            verdict = "weapon"
            reasoning = f"Weapon detected with {confidence:.1%} confidence. Contains {ocr_text if ocr_text else 'text/calibration marks'}."
            action = "flag" if confidence > 0.6 else "review"

        elif "product" in class_name:
            verdict = "counterfeit" if confidence > 0.7 else "product"
            reasoning = f"Product detected with {confidence:.1%} confidence. Requires manual review."
            action = "review"

        else:
            verdict = "safe"
            reasoning = f"Object detected but not a known violation. Confidence: {confidence:.1%}"
            action = "allow"

        return {
            "violation_type": verdict,
            "confidence": min(confidence, 0.95),
            "reasoning": reasoning,
            "evidence": [class_name, f"confidence:{confidence:.2f}"],
            "recommended_action": action,
            "method": "rule_based"
        }

    def reason_about_detection(self, detection: Dict, ocr_text: Optional[str] = None) -> Dict:
        """Generate reasoning verdict for detection"""
        logger.info(f"🧠 Generating reasoning for {detection.get('class_name', 'unknown')}...")

        # If model not available, use rule-based
        if not self.model:
            verdict = self.rule_based_reasoning(detection, ocr_text)
            logger.info(f"   (Rule-based) Verdict: {verdict['violation_type']}")
            return verdict

        # Try LLaVA reasoning
        try:
            prompt = self.generate_reasoning_prompt(detection, ocr_text)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON from response
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                verdict = json.loads(json_str)
                logger.info(f"   ✓ Verdict: {verdict['violation_type']}")
                return verdict
            except:
                logger.warning("   ⚠ Could not parse JSON response")
                return self.rule_based_reasoning(detection, ocr_text)

        except Exception as e:
            logger.warning(f"   ⚠ LLaVA reasoning failed: {e}")
            return self.rule_based_reasoning(detection, ocr_text)

    def analyze_phase3_results(self, phase3_file: Path) -> List[Dict]:
        """Process Phase 3 results and add reasoning"""
        logger.info(f"\n📂 Loading Phase 3 results: {phase3_file.name}")

        with open(phase3_file, 'r') as f:
            phase3_results = json.load(f)

        logger.info(f"✓ Loaded {len(phase3_results)} images\n")

        reasoning_results = []

        for idx, result in enumerate(phase3_results, 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"IMAGE {idx}/{len(phase3_results)}")
            logger.info(f"{'=' * 70}")

            file_name = Path(result['file']).name
            logger.info(f"File: {file_name}")

            detections = result.get('detections', [])
            ocr_results = result.get('ocr', [])

            if not detections:
                logger.info("  No detections found")
                reasoning_results.append({
                    "file": result['file'],
                    "detections": [],
                    "reasoning": []
                })
                continue

            # Generate reasoning for each detection
            reasoning = []
            for det in detections:
                ocr_text = None
                if ocr_results:
                    ocr_text = " ".join([o['text'] for o in ocr_results[:3]])

                verdict = self.reason_about_detection(det, ocr_text)
                reasoning.append(verdict)

            reasoning_results.append({
                "file": result['file'],
                "detections": detections,
                "ocr": ocr_results,
                "reasoning": reasoning
            })

        return reasoning_results


def main():
    """Run Phase 4 VLM reasoning"""

    # Initialize VLM engine
    engine = VLMReasoningEngine()

    # Load Phase 3 results
    phase3_file = Path("results/phase3_analysis/phase3_results.json")

    if not phase3_file.exists():
        logger.error(f"❌ Phase 3 results not found: {phase3_file}")
        logger.info("   Run Phase 3 first: python scripts/phase3_ocr_asr_pipeline.py")
        return

    # Generate reasoning
    logger.info("\n" + "=" * 70)
    logger.info("ANALYZING WITH VLM REASONING")
    logger.info("=" * 70 + "\n")

    reasoning_results = engine.analyze_phase3_results(phase3_file)

    # Save results
    output_dir = Path("results/phase4_reasoning")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "phase4_verdicts.json"
    with open(output_file, 'w') as f:
        json.dump(reasoning_results, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("PHASE 4 COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"✅ Analyzed {len(reasoning_results)} images")
    logger.info(f"✅ Verdicts saved: {output_file}\n")

    # Summary statistics
    total_violations = sum(len(r['detections']) for r in reasoning_results)
    flagged = sum(1 for r in reasoning_results for v in r['reasoning'] if v['recommended_action'] == 'flag')

    logger.info("📊 STATISTICS:")
    logger.info(f"   Total violations detected: {total_violations}")
    logger.info(f"   Violations flagged: {flagged}")
    logger.info(f"   Average confidence: {sum(v['confidence'] for r in reasoning_results for v in r['reasoning']) / max(1, sum(len(r['reasoning']) for r in reasoning_results)):.1%}")

    logger.info(f"\n🚀 Next: Phase 5 - Metrics Pipeline & Ablation Studies")


if __name__ == "__main__":
    import torch
    main()
