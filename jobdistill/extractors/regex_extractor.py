"""Legacy regex-based skill extractor (lifted from original main.py)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from jobdistill.extractors.base import ExtractionResult, SkillExtractor


def get_possible_skills() -> List[str]:
    """The original hardcoded skills inventory.

    Exposed as a function so scripts/build_training_data.py can import it
    as a weak-label seed list without using it for inference.
    """
    return [
        "C++", "C", "C#", "Python", "Pandas", "AWS", "Java", "JavaScript", "Go", "SQL",
        "REST", "TypeScript", "React", "Node", "API", "Excel", "Linux", "Agile", "Cloud",
        "ML", "Oracle", "Scrum", "Jira", "Windows", "Docker", ".NET", "Word", "Shell",
        "Ruby", "Perl", "Kubernetes", "Unix", "Selenium", "Visual Studio", "VS Code",
        "Swift", "Scala", "Hadoop", "Git", "GitHub", "Tableau", "PowerBI",
        "Microsoft Office", "Jenkins", "TensorFlow", "Matlab", "CI/CD", "Django",
        "Google Cloud", "Robot Framework", "Cloud Computing", "Azure", "Jupyter",
        "Salesforce", "Keras", "Kafka", "PyTorch", "Figma", "Angular", "Assembly",
        "Arduino", "Unreal Engine", "Big Data", "Rust", "Flask", "Vue", "Matplotlib",
        "Objective-C", "Anaconda", "Redis", "CLI", "Vercel", "BitBucket", "npm", "PNPM",
        "Remix", "Tanstack", "HTML", "CSS", "ShadCN", "RadixUI", "MUI", "TCP/IP", "gRPC",
        "HTTP", "TLS", "Web Sockets", "GitHub Actions", "Springboot", "DevOps", "SDLC",
        "VBA", "Pyautogui", "Razor", "Xamarin", "Maui", "Controller Area Network",
        "ISOBUS", "RS232", "Unit Testing", "JSON", "Simulink", "LiDAR", "Apache Spark",
        "Apache Hop", "Terraform", "Databricks", "Unity", "Object-Oriented",
        "Data Structure", "XML", "XSD", "SOAP", "Eclipse", "SuiteBuilder", "NetSuite",
        "SuiteBundler", "SuiteFlow", "SuiteTalk", "Celigo", "Integrator.io", "Dell Boomi",
        "SPS", "Spring Boot", "Kotlin", "R", "AutoCAD", "SharePoint", "Dojo", "Groovy",
        "Artifactory", "IBM Cloud", "NoSQL", "MySQL", "Db2", "Cloudant", "OpenShift",
        "Ansible", "Cypress", "ngrx", "RxJS", "GraphQL", "Apollo", "RTOS", "Ethernet",
        "SPI", "I2C", "ARM", "ATOM", "DSP", "FPGA", "GPU", "LIN", "UDS", "CANApe",
        "UIKit", "LLMs", "Hugging Face", "OpenCV", "AI", "SDK", "Xcode",
        "Android Studio", "PostgreSQL", "MongoDB", "Rails", "XSL", "Perforce", "OAuth",
        "SSL", "NDK", "JNI", "Jetpack", "DQN", "LSTM", "SVM", "random forest", "VFX",
        "OpenGL", "WebGL", "DirectX", "Metal", "Vulkan", "Verilog", "SystemVerilog",
        "VHDL", "Digital Signal Processing", "bash", "Tcl", "LangChain", "Redux", "Helm",
        "LabVIEW", "JQuery", "GitLab", "Julia", "NLP", "scikit-learn", "sklearn", "GUI",
        "DAST", "SAST", "SCA", "CVE", "CVSS", "CWE", "ETL", "PowerPoint", "Kanban",
        "Playwright", "Postman", "TeamCity", "OpenAI", "Sagemaker", "Metaflow", "YAML",
        "SCM", "JTAG", "PCIe", "CXL", "FME Flow Hosted", "FME Hub", "FME Account",
        "FME AI Service", "Spring", "FastAPI", "Apache Tomcat", "JBoss", "Websocket",
        "JPA2", "ORM", "Hibernate", "Cassandra", "HBase", "Flink", "cmake", "5G",
        "TestCafe", "Computer Vision Algorithms", "Networking Protocols", "Multithreading",
        "Outlook", "Gerrit", "S3", "Microservices", "T-SQL", "Knockout", "JSCON", "OData",
        "ERD", "SSRS", "Flutter", "Express", "GenAI", "Bamboo", "Apex", "CRM", "SaaS",
        "SAP Analytics", "DNS", "Endpoint Protection", "Web Proxy",
        "Vulnerability Management", "SIEM", "Access", "BigQuery", "Compute Engine",
        "XUnit", "PHP", "IoT", "Raspberry Pi", "Pub/Sub", "asyncio", "VMWare", "QEMU",
        "Security Operations", "Threat Containment", "Zero Trust", "SVN", "MS Build",
        "Optical Fibers", "Open RAN", "Elixir", "RabbitMQ", "LAN", "WAN", "MS Visio",
        "D3", "Vite", "Rush", "Flux", "ECMAScript", "K8S", "Ember", "CCNA", "Bootstrap",
        "IQR", "K-S test", "Regression", "ARIMA", "DHCP", "VLANS", "OSPF", "BGP", "ESXi",
        "Xen", "InfluxDB", "Multi-tiered Systems", "Integration Testing",
        "Relational Databases", "Linear Programming", "Electron", "UDP", "OpenStack",
        "PKI", "Vagrant", "Gradle", "RF Test", "ARP", "Packer", "Cloud-Init", "Kickstart",
        "Blockchain", "JAAS", "SuiteScript", "React Native", "OpenCL", "Direct3D", "UML",
        "Trello", "Adobe Creative", "Metabase", "Next.js", "Comment Standard",
        "Argo Workflows", "particle.js", "ROS", "Gazebo", "BuildBot", "Solidity",
        "TailwindCSS", "SMACSS", "Tauri", "Astro", "SvelteKit", "Qwik", "MobX", "SolidJS",
        "SAML", "Kerberos", "SSO", "ABAP", "SAP HANA", "Nuxt 3", "Firebase", "Firestore",
        "Web3", "Gatsby", "VIM", "ThreeJS", "GLSL", "JAMStack", "Elm", "Webpack",
        "LevelDB", "Discord.js", "Star Schema", "Waterfall", "Hive", "Beeline", "HDFS",
        "Sqoop", "MinIO", "Airflow", "Trino", "PyHive", "PySpark", "Korn",
    ]


def get_special_regex_skills() -> Dict[str, str]:
    """Regex overrides for short / ambiguous skill names."""
    return {
        "C": r"\bC(?!\+)\b|\bC programming\b|\bC language\b|\bC developer\b",
        "R": r"\bR\b|\bR programming\b|\bR language\b|\bR developer\b",
        "ARM": r"\bARM\b|\bARM processor\b|\bARM architecture\b",
        "API": r"\bAPI\b|\bAPIs\b|\bREST API\b|\bWeb API\b",
        "GUI": r"\bGUI\b|\bGUIs\b|\buser interface\b",
        "SQL": r"\bSQL\b|\bMySQL\b|\bPostgreSQL\b|\bSQL Server\b|\bSQL database\b",
        "XML": r"\bXML\b|\bXML format\b|\bXML document\b",
        "SDK": r"\bSDK\b|\bSDKs\b|\bsoftware development kit\b",
        "UDS": r"\bUDS\b|\bUDS protocol\b",
        "AI": r"\bAI\b",
        "ML": r"\bML\b",
        "UI": r"\bUI\b|\bUI/UX\b|\buser interface\b",
        "SPI": r"\bSPI\b|\bSPI protocol\b|\bSerial Peripheral Interface\b",
        "I2C": r"\bI2C\b|\bI2C protocol\b|\bI²C\b",
        "CLI": r"\bCLI\b|\bcommand[ -]line interface\b|\bcommand[ -]line tool\b",
        "Java": r"\bJava\b",
        "Elm": r"\bElm\b",
        "Helm": r"\bHelm\b",
    }


def get_skill_mapping() -> Dict[str, object]:
    """Alias -> canonical form mapping. Values may be str or List[str]."""
    return {
        "Javascript": "JavaScript", "golang": "Go", "Golang": "Go",
        "Go (golang)": "Go", "GO": "Go", "C/C++": ["C", "C++"],
        "AI/ML": ["AI", "ML"], "Node.js": "Node", "NodeJS": "Node",
        "React.js": "React", "ReactJS": "React", "Typescript": "TypeScript",
        "CICD": "CI/CD", "Rest": "REST", "RESTful": "REST",
        "REST API": "REST", "REST APIs": "REST", ".Net": ".NET",
        "ASP.NET": ".NET", "VB.NET": ".NET", "VB": ".NET", "ASP": ".NET",
        "Net": ".NET", "MS Office": "Microsoft Office",
        "MS Suite": "Microsoft Office", "Office 365": "Microsoft Office",
        "Microsoft Word": "Word", "MS Word": "Word",
        "Microsoft Excel": "Excel", "MS Excel": "Excel",
        "Machine Learning": "ML", "machine learning": "ML",
        "Artifical Intelligence": "AI", "artifical intelligence": "AI",
        "Large Language Models": "LLMs", "LMMS": "LLMs", "pandas": "Pandas",
        "HTML5": "HTML", "CSS3": "CSS", "GIT": "Git", "git": "Git",
        "Github": "GitHub", "Bitbucket": "BitBucket", "MATLAB": "Matlab",
        "Pytorch": "PyTorch", "Tensorflow": "TensorFlow", "Vue.Js": "Vue",
        "Vue.js": "Vue", "VUE": "Vue", "HTML/CSS": ["HTML", "CSS"],
        "CSS/SCSS": "CSS", "PyTest": "Unit Testing", "JUnit": "Unit Testing",
        "GoogleTest": "Unit Testing", "Object-Oriented": "OOP",
        "object-oriented": "OOP", "NPM": "npm", "DSA": "Data Structure",
        "data structure": "Data Structure",
        "complexity analysis": "Data Structure",
        "operating systems": "OS", "Operating Systems": "OS",
        "Linux/Unix": "Linux", "Rabbit": "RabbitMQ", "JAVA": "Java",
        "Power BI": "PowerBI", "PowerShell": "Shell", "Postgres": "PostgreSQL",
        "DEVOPS": "DevOps",
        "computer vision algorithms": "Computer Vision Algorithms",
        "networking protocols": "Networking Protocols",
        "multithreaded": "Multithreading", "multi-threading": "Multithreading",
        "JPA": "JPA2", "DB2": "Db2", "microservices": "Microservices",
        "Micro-Services": "Microservices", "Micro-services": "Microservices",
        "optical fibers": "Optical Fibers", "JIRA": "Jira",
        "S3": "Amazon S3", "Amazon Web Services": "AWS", "flux": "Flux",
        "RushJs": "Rush", "RushJS": "Rush", "Rush.js": "Rush",
        "GCP": "Google Cloud", "Google Cloud Platform": "Google Cloud",
        "bootstrap": "Bootstrap", "regression": "Regression",
        "Express.js": "Express", "blockchain": "Blockchain",
        "linear programming": "Linear Programming",
        "multi-tiered systems": "Multi-tiered Systems",
        "Rx.js": "RxJS", "RxJS": "RxJS", "rxjs": "RxJS",
        "relational databases": "Relational Databases",
        "relational database": "Relational Databases",
        "Relational Database": "Relational Databases",
    }


def _compile_patterns(
    skills: List[str], special: Dict[str, str]
) -> Dict[str, "re.Pattern[str]"]:
    compiled: Dict[str, re.Pattern[str]] = {}
    for skill in skills:
        if skill in special:
            pattern = special[skill]
        elif re.match(r"^[a-zA-Z]", skill) and not re.search(r"\W", skill):
            pattern = r"\b" + re.escape(skill) + r"\b"
        else:
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        compiled[skill] = re.compile(pattern, re.IGNORECASE)
    return compiled


class RegexSkillExtractor(SkillExtractor):
    """Drop-in replacement for the original main.py extraction logic."""

    def __init__(self) -> None:
        self._skills = get_possible_skills()
        self._special = get_special_regex_skills()
        self._mapping = get_skill_mapping()
        self._patterns = _compile_patterns(self._skills, self._special)

    @property
    def name(self) -> str:
        return "regex"

    def extract(self, text: str) -> ExtractionResult:
        counts = self._count_skills(text)
        skills_dict = {s: 1.0 for s, _ in counts.items() if counts[s] > 0}
        return ExtractionResult(
            skills=skills_dict,
            candidates_considered=len(self._skills),
        )

    def extract_counts(self, text: str) -> Counter:
        """Return raw skill -> count (for pipeline aggregation)."""
        return self._count_skills(text)

    def _count_skills(self, text: str) -> Counter:
        skill_counts: Counter = Counter()

        for skill, pattern in self._patterns.items():
            count = len(pattern.findall(text))
            if count > 0:
                skill_counts[skill] = count

        for alias, normalized in self._mapping.items():
            normalized_skills = normalized if isinstance(normalized, list) else [normalized]
            if re.match(r"^[a-zA-Z]", alias) and not re.search(r"\W", alias):
                alias_pattern = r"\b" + re.escape(alias) + r"\b"
            else:
                alias_pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            alias_count = len(re.findall(alias_pattern, text, re.IGNORECASE))
            if alias_count > 0:
                for norm_skill in normalized_skills:
                    if norm_skill in self._patterns:
                        skill_counts[norm_skill] += alias_count

        return skill_counts
