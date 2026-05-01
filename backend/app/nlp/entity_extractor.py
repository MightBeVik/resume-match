"""Entity extraction using spaCy NER, POS tagging, and custom patterns."""
import re
from functools import lru_cache

import spacy

from app import models_state

# ---------------------------------------------------------------------------
# Known skills — matched against both resumes and job descriptions.
# Add new skills here (lowercase); add display overrides in DISPLAY_OVERRIDES.
# ---------------------------------------------------------------------------
KNOWN_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    "bash", "powershell", "shell",
    # Frontend
    "react", "angular", "vue", "svelte", "next.js", "nuxt", "astro",
    "html", "css", "sass", "tailwind", "bootstrap", "webpack", "vite",
    "redux", "zustand", "graphql", "grpc",
    # Backend / Frameworks
    "node.js", "nodejs", "express", "django", "flask", "fastapi", "spring",
    "nestjs", "fastify", "gin", "echo", "rails", "laravel", "symfony",
    "asp.net", "dotnet", ".net",
    # Databases
    "sql", "nosql", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "cassandra", "dynamodb", "firestore", "neo4j",
    "snowflake", "bigquery", "redshift", "supabase",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "helm", "pulumi", "jenkins", "github actions", "gitlab ci", "ci/cd",
    "linux", "nginx", "apache", "devops", "sre",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "data science",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "xgboost", "lightgbm", "spark", "kafka", "airflow",
    "dbt", "databricks", "mlflow", "kubeflow", "langchain", "llamaindex",
    "openai", "hugging face", "transformers",
    # BI / Analytics
    "tableau", "power bi", "looker", "metabase", "dax", "excel",
    "powerpoint", "google analytics",
    # ORMs / Data tools
    "prisma", "sqlalchemy", "sequelize", "typeorm", "hibernate",
    # Testing
    "jest", "pytest", "cypress", "selenium", "playwright",
    "junit", "mocha", "chai",
    # Version control / Collab
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "agile", "scrum", "kanban",
    # Other platforms
    "salesforce", "sap", "oracle", "figma", "sketch", "adobe",
    "photoshop", "rest", "rest apis", "microservices",
    # SaaS APIs
    "stripe", "twilio", "sendgrid", "firebase",
}

# ---------------------------------------------------------------------------
# Known professional certifications
# ---------------------------------------------------------------------------
KNOWN_CERTIFICATIONS = {
    # Cloud
    "aws certified", "aws solutions architect", "aws developer",
    "aws sysops", "aws devops professional",
    "google cloud certified", "gcp professional", "google cloud associate",
    "azure certified", "az-900", "az-104", "az-204", "az-305",
    # Security
    "cissp", "cism", "cisa", "comptia security+", "comptia network+",
    "comptia a+", "ceh", "oscp", "ccna", "ccnp", "ccsp",
    # Project / Agile
    "pmp", "prince2", "pmi-acp", "csm", "scrum master", "safe",
    "certified scrum master",
    # Data / AI
    "tensorflow developer", "google ml engineer",
    "databricks certified", "snowflake certified",
    # Finance / Business
    "cpa", "cfa", "cga", "cma", "frm", "acca",
    # Healthcare
    "phr", "sphr", "shrm",
    # Other
    "itil", "six sigma", "lean six sigma", "togaf",
}

DEGREE_PATTERNS = re.compile(
    r"(?:bachelor(?:\'s)?|master(?:\'s)?|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|"
    r"ph\.?d\.?|mba|associate(?:\'s)?|doctorate)\s*(?:of\s+(?:science|arts|engineering|business))?"
    r"(?:\s+in\s+[\w\s&]+)?",
    re.IGNORECASE,
)

DISPLAY_OVERRIDES = {
    # --- original entries ---
    "agile": "Agile",
    "aws": "AWS",
    "azure": "Azure",
    "bash": "Bash",
    "c#": "C#",
    "c++": "C++",
    "ci/cd": "CI/CD",
    "css": "CSS",
    "devops": "DevOps",
    "django": "Django",
    "docker": "Docker",
    "excel": "Excel",
    "fastapi": "FastAPI",
    "figma": "Figma",
    "gcp": "GCP",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "go": "Go",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "html": "HTML",
    "java": "Java",
    "javascript": "JavaScript",
    "jenkins": "Jenkins",
    "jira": "Jira",
    "kubernetes": "Kubernetes",
    "linux": "Linux",
    "machine learning": "Machine Learning",
    "microservices": "Microservices",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "nlp": "NLP",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "nosql": "NoSQL",
    "numpy": "NumPy",
    "oracle": "Oracle",
    "pandas": "Pandas",
    "php": "PHP",
    "postgresql": "PostgreSQL",
    "power bi": "Power BI",
    "powerpoint": "PowerPoint",
    "python": "Python",
    "pytorch": "PyTorch",
    "react": "React",
    "redis": "Redis",
    "rest": "REST",
    "rest apis": "REST APIs",
    "ruby": "Ruby",
    "rust": "Rust",
    "salesforce": "Salesforce",
    "sap": "SAP",
    "sass": "Sass",
    "scikit-learn": "scikit-learn",
    "scrum": "Scrum",
    "sketch": "Sketch",
    "sql": "SQL",
    "swift": "Swift",
    "tableau": "Tableau",
    "tailwind": "Tailwind",
    "tensorflow": "TensorFlow",
    "terraform": "Terraform",
    "typescript": "TypeScript",
    "vite": "Vite",
    "vue": "Vue",
    "webpack": "Webpack",
    # --- new entries ---
    ".net": ".NET",
    "airflow": "Airflow",
    "ansible": "Ansible",
    "apache": "Apache",
    "asp.net": "ASP.NET",
    "astro": "Astro",
    "aws certified": "AWS Certified",
    "bigquery": "BigQuery",
    "bitbucket": "Bitbucket",
    "cassandra": "Cassandra",
    "chai": "Chai",
    "cissp": "CISSP",
    "cpa": "CPA",
    "cfa": "CFA",
    "cypress": "Cypress",
    "databricks": "Databricks",
    "dax": "DAX",
    "dbt": "dbt",
    "deep learning": "Deep Learning",
    "dotnet": ".NET",
    "dynamodb": "DynamoDB",
    "echo": "Echo (Go)",
    "elasticsearch": "Elasticsearch",
    "fastify": "Fastify",
    "firebase": "Firebase",
    "firestore": "Firestore",
    "gcp professional": "GCP Professional",
    "gin": "Gin (Go)",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "google analytics": "Google Analytics",
    "google cloud certified": "Google Cloud Certified",
    "helm": "Helm",
    "hibernate": "Hibernate",
    "hugging face": "Hugging Face",
    "junit": "JUnit",
    "kafka": "Kafka",
    "kanban": "Kanban",
    "keras": "Keras",
    "kotlin": "Kotlin",
    "kubeflow": "Kubeflow",
    "langchain": "LangChain",
    "laravel": "Laravel",
    "lightgbm": "LightGBM",
    "llamaindex": "LlamaIndex",
    "looker": "Looker",
    "matlab": "MATLAB",
    "metabase": "Metabase",
    "mlflow": "MLflow",
    "mocha": "Mocha",
    "neo4j": "Neo4j",
    "nestjs": "NestJS",
    "next.js": "Next.js",
    "nginx": "Nginx",
    "nuxt": "Nuxt",
    "openai": "OpenAI",
    "playwright": "Playwright",
    "pmp": "PMP",
    "powershell": "PowerShell",
    "prisma": "Prisma",
    "pulumi": "Pulumi",
    "pytest": "Pytest",
    "rails": "Rails",
    "r": "R",
    "redshift": "Redshift",
    "redux": "Redux",
    "scala": "Scala",
    "selenium": "Selenium",
    "sendgrid": "SendGrid",
    "sequelize": "Sequelize",
    "shell": "Shell",
    "snowflake": "Snowflake",
    "spark": "Spark",
    "sqlalchemy": "SQLAlchemy",
    "sqlite": "SQLite",
    "sre": "SRE",
    "stripe": "Stripe",
    "supabase": "Supabase",
    "svelte": "Svelte",
    "symfony": "Symfony",
    "transformers": "Transformers",
    "twilio": "Twilio",
    "typeorm": "TypeORM",
    "xgboost": "XGBoost",
    "zustand": "Zustand",
}


def _get_nlp():
    """Get spaCy model, preferring the preloaded one."""
    if models_state.nlp_model is not None:
        return models_state.nlp_model
    return spacy.load("en_core_web_sm")


def _format_skill(skill: str) -> str:
    normalized = skill.strip().lower()
    return DISPLAY_OVERRIDES.get(normalized, skill.strip())


@lru_cache(maxsize=None)
def _compile_skill_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def _contains_skill(text: str, skill: str) -> bool:
    return bool(_compile_skill_pattern(skill).search(text))


def extract_skills(text: str) -> list[str]:
    """Extract technical skills using NLP and known skill matching.

    Uses: POS tagging, noun phrase chunking, pattern matching.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    found_skills = set()

    # Method 1: Match against known skills list (longest match first to prefer
    # "machine learning" over just "machine")
    text_lower = text.lower()
    for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
        if _contains_skill(text_lower, skill):
            found_skills.add(_format_skill(skill))

    # Method 2: Extract noun phrases (noun phrase chunking)
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower().strip()
        if chunk_lower in KNOWN_SKILLS:
            found_skills.add(_format_skill(chunk_lower))

    # Method 3: Look for comma-separated lists after skill-related headers
    skill_list_pattern = re.compile(
        r"(?:skills|technologies|tech stack|tools|frameworks|languages)\s*:?\s*(.+?)(?:\n\n|\n[A-Z]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in skill_list_pattern.finditer(text):
        items = re.split(r"[,\|;]", match.group(1))
        for item in items:
            cleaned = item.strip().strip("-•* ")
            if cleaned and 1 < len(cleaned) < 40:
                if cleaned.lower() in KNOWN_SKILLS:
                    found_skills.add(_format_skill(cleaned))
                elif cleaned[0].isupper() and len(cleaned.split()) <= 3:
                    found_skills.add(cleaned.strip())

    return sorted({_format_skill(skill) for skill in found_skills}, key=str.lower)


def extract_certifications(text: str) -> list[str]:
    """Extract professional certifications from text.

    Matches against KNOWN_CERTIFICATIONS and common certification
    phrase patterns (e.g. "Certified in X", "X Certification").
    """
    found: set[str] = set()
    text_lower = text.lower()

    # Method 1: Known certification list (longest first for specificity)
    for cert in sorted(KNOWN_CERTIFICATIONS, key=len, reverse=True):
        if cert in text_lower:
            # Title-case for display
            found.add(cert.title())

    # Method 2: Regex patterns — "Certified X", "X Certification", "X Certificate"
    cert_patterns = [
        re.compile(
            r"(?:certified\s+in\s+)([\w\s/&+#.-]{3,40}?)(?=\s*[,.\n]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"([\w\s/&+#.-]{3,40}?)\s+(?:certification|certificate|credential)(?=\s*[,.\n]|$)",
            re.IGNORECASE,
        ),
    ]
    for pattern in cert_patterns:
        for match in pattern.finditer(text):
            candidate = match.group(1).strip()
            if 3 < len(candidate) < 60 and not re.search(r"\d{4}", candidate):
                found.add(candidate)

    return sorted(found)


def extract_education(text: str) -> list[str]:
    """Extract education items using regex patterns and NER."""
    found = set()

    for match in DEGREE_PATTERNS.finditer(text):
        degree = match.group(0).strip()
        if len(degree) > 3:
            found.add(degree)

    return sorted(found)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract all entity types from text.

    Returns dict with keys: skills, certifications, organizations, education.
    Uses: spaCy NER, POS tagging, noun phrase chunking, regex patterns.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    organizations = set()
    for ent in doc.ents:
        cleaned = ent.text.strip()
        if ent.label_ == "ORG" and cleaned and "," not in cleaned and cleaned.lower() not in KNOWN_SKILLS:
            organizations.add(ent.text.strip())

    return {
        "skills": extract_skills(text),
        "certifications": extract_certifications(text),
        "organizations": sorted(organizations),
        "education": extract_education(text),
    }
