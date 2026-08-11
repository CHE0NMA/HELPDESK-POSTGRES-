"""
database/learning_schema.py — boot-time migrations for the continuous-learning
layer (topic aliases, feedback, success cache, optional chunk tags).

Call ensure_learning_schema(get_db) once from create_app() after ensure_core_tables().
Then seed_default_aliases(get_db) so ranking works before any admin edits.
"""
import logging

logger = logging.getLogger("pac_helpdesk.learning_schema")


def ensure_learning_schema(get_db_fn) -> None:
    try:
        from services.topic_aliases import ensure_topic_alias_schema, seed_default_aliases
    except ImportError:
        try:
            from topic_aliases import ensure_topic_alias_schema, seed_default_aliases
        except ImportError as e:
            logger.error(f"topic_aliases import failed: {e}")
            return

    ensure_topic_alias_schema(get_db_fn)
    n = seed_default_aliases(get_db_fn, force=False)
    if n:
        logger.info(f"Seeded {n} handbook topic aliases")


def wire_into_app(app, get_db_fn, scheduler=None):
    """
    Optional helper: call from create_app after scheduler starts.

    - Ensures schema + seeds
    - Schedules weekly alias suggestions log (admin can read via API)
    """
    ensure_learning_schema(get_db_fn)

    if scheduler is not None:
        try:
            from services.topic_aliases import suggest_aliases_from_unanswered

            def _log_suggestions():
                try:
                    tips = suggest_aliases_from_unanswered(get_db_fn, min_count=3, limit=15)
                    if tips:
                        logger.info(
                            "Alias suggestions (%d): %s",
                            len(tips),
                            "; ".join(t["sample_question"][:40] for t in tips[:5]),
                        )
                except Exception as e:
                    logger.warning(f"alias suggestion job: {e}")

            scheduler.add_job(func=_log_suggestions, trigger="interval", hours=24)
        except Exception as e:
            logger.warning(f"Could not schedule alias suggestions: {e}")
