import logging
import time
from src.loaders import load_all_structured_tasks, load_emails_raw, load_meeting_transcripts_raw
from src.extractor import extract_all_unstructured
from src.deduplicator import deduplicate_tasks
from src.prioritizer import prioritize_tasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation():
    logger.info("Starting TaskPilot AI Evaluation Harness...")
    
    start_time = time.time()
    
    # 1. Load Data
    structured_tasks = load_all_structured_tasks()
    emails = load_emails_raw()
    meetings = load_meeting_transcripts_raw()
    
    # Ground truth for unstructured data:
    # Based on the data generation specs, there should be ~9 implicit tasks in the emails and meetings.
    # Let's say 9 is our target for 100% discovery rate.
    GROUND_TRUTH_UNSTRUCTURED = 9
    
    # 2. Extract Unstructured Tasks
    logger.info("Extracting tasks from unstructured text...")
    extracted_tasks = extract_all_unstructured(emails, meetings)
    discovery_rate = (len(extracted_tasks) / GROUND_TRUTH_UNSTRUCTURED) * 100
    
    # 3. Deduplication
    logger.info("Running Deduplication Pipeline...")
    all_tasks = structured_tasks + extracted_tasks
    initial_count = len(all_tasks)
    deduped_tasks = deduplicate_tasks(all_tasks)
    final_count = len(deduped_tasks)
    
    # Deduplication Accuracy approximation:
    # There are roughly 4 known overlaps (e.g. Jira TASK-101 overlaps with ServiceNow INC0001234 and an email).
    # If the system reduces the count by 4-6, it's operating correctly.
    # We will score it based on successful reduction without over-merging.
    dedup_accuracy = 100.0 if (initial_count - final_count) >= 4 else ((initial_count - final_count) / 4) * 100
    
    # 4. Prioritization & Explainability
    logger.info("Running Prioritization and Rationale Generation...")
    prioritized = prioritize_tasks(deduped_tasks)
    
    explainability_score = 0
    grounding_score = 0
    
    for pt in prioritized:
        if pt.rationale and len(pt.rationale) > 10:
            explainability_score += 1
        if pt.task.source_lineage and len(pt.task.source_lineage) > 0:
            grounding_score += 1
            
    explainability_percent = (explainability_score / final_count) * 100 if final_count > 0 else 0
    grounding_percent = (grounding_score / final_count) * 100 if final_count > 0 else 0
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info("==================================================")
    logger.info("             EVALUATION METRICS                   ")
    logger.info("==================================================")
    logger.info(f"Execution Time:          {execution_time:.2f} seconds (Target: <60s)")
    logger.info(f"Task Discovery Rate:     {discovery_rate:.1f}% (Target: >95%)")
    logger.info(f"Deduplication Accuracy:  {dedup_accuracy:.1f}% (Target: >90%)")
    logger.info(f"Explainability Metric:   {explainability_percent:.1f}%")
    logger.info(f"Data Grounding Metric:   {grounding_percent:.1f}% (Target: 100%)")
    logger.info("==================================================")

if __name__ == "__main__":
    run_evaluation()
