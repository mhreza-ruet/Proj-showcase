def looks_like_course_query(norm_q: str) -> bool:
    return any(
        k in norm_q
        for k in [
            "quiz", "quizzes",
            "hw", "homework",
            "assignment", "assignments",
            "deadline", "deadlines",
            "due", "due date", "due dates",
            "grading", "grading policy",
            "creation",
            "evaluation",
            "project", "projects",
            "lab", "labs",
            "exam", "exams",
            "kritik",
            "task", "tasks",
            "submission", "submissions",
            "submit",
            "deliverable", "deliverables",
            "recording", "recordings",
            "slides",
            "policy", "policies",
            "announcement", "announcements",
            "latest", "last", "recent", "newest", "upcoming",
        ]
    )


def choose_retrieve_mode(latest_query: bool, announcement_priority: bool, deadline_query: bool) -> str:
    if latest_query and (announcement_priority or deadline_query):
        return "announcements_only_recent"
    elif announcement_priority or deadline_query or latest_query:
        return "announcements_first"
    else:
        return "mixed"