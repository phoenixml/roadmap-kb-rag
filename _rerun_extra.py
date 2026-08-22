"""Run additional (weaker) models through the corrected retrieval pipeline.
Writes to a separate file; reports honestly whatever comes out."""
import sys
sys.path.insert(0, ".")
import _rerun_fixed as R

R.MODELS = {
    "gpt-3.5-turbo":     {"provider": "openai",    "model": "gpt-3.5-turbo"},
    "claude-haiku-4-5":  {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
}
R.OUT_JSON = "outputs/qa_eval_results_atlas_extra.json"
R.OUT_CSV  = "outputs/qa_eval_summary_atlas_extra.csv"

if __name__ == "__main__":
    print("[extra] models:", list(R.MODELS.keys()))
    R.main()
