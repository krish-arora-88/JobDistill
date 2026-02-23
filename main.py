import glob
import os
import re
import logging
import concurrent.futures
import gc
import argparse
from tqdm import tqdm
from collections import Counter
from pdfminer.high_level import extract_text
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='skill_analysis.log')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze skills in job postings')
    parser.add_argument('--pdf_dirs', nargs='+', default=['Summer_2025_Co-op', 'Fall_2025_Co-op', 'Winter_2026_Co-op', 'Summer_2026_Co-op'],
                        help='Directories containing PDF files')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of PDFs to process in each batch')
    return parser.parse_args()

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file with error handling."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def remove_duplicates_in_text(text):
    """
    Remove duplicate words from the text, keeping only
    the first occurrence of each token.
    """
    # Simple split on whitespace. Adjust if you want more robust tokenization.
    tokens = text.split()
    seen = set()
    unique_tokens = []

    for token in tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    return " ".join(unique_tokens)

def create_skill_mapping():
    """Create mapping dictionary to normalize skill names."""
    return {
        # JavaScript ecosystem
        "Javascript": "JavaScript",
        "golang": "Go",
        "Golang": "Go",
        "Go (golang)": "Go",
        "GO": "Go",
        
        "C/C++": ["C", "C++"],
        
        "AI/ML": ["AI", "ML"],
        
        # JavaScript variations
        "Javascript": "JavaScript",
        "Node.js": "Node",
        "NodeJS": "Node",
        "React.js": "React",
        "ReactJS": "React",
        
        # TypeScript variations
        "Typescript": "TypeScript",
        
        # CI/CD variations
        "CICD": "CI/CD",
        
        # REST variations
        "Rest": "REST",
        "RESTful": "REST",
        "REST API": "REST",
        "REST APIs": "REST",
        
        # .NET variations
        ".Net": ".NET",
        "ASP.NET": ".NET",
        "VB.NET": ".NET",
        "VB": ".NET",
        "ASP": ".NET",
        "Net": ".NET",
        
        # Office variations
        "MS Office": "Microsoft Office",
        "MS Suite": "Microsoft Office",
        "Office 365": "Microsoft Office",
        "Microsoft Word": "Word",
        "MS Word": "Word",
        "Microsoft Excel": "Excel",
        "MS Excel": "Excel",
        
        # ML/AI variations
        "Machine Learning": "ML",
        "machine learning": "ML",
        "Artifical Intelligence": "AI",
        "artifical intelligence": "AI",
        "Large Language Models": "LLMs",
        "LMMS": "LLMs",
        
        # Python libraries that could be grouped
        "pandas": "Pandas",
        
        # Version-specific technologies
        "HTML5": "HTML",
        "CSS3": "CSS",
        
        # Git variations
        "GIT": "Git",
        "git": "Git",
        "Github": "GitHub",
        "Bitbucket": "BitBucket",
        
        # Case variations for other tools
        "MATLAB": "Matlab",
        "Pytorch": "PyTorch",
        "Tensorflow": "TensorFlow",
        
        # Web framework variations
        "Vue.Js": "Vue",
        "Vue.js": "Vue",
        "VUE": "Vue",
        
        # HTML/CSS combinations
        "HTML/CSS": ["HTML", "CSS"],
        "CSS/SCSS": "CSS",
        
        # Testing frameworks
        "PyTest": "Unit Testing",
        "JUnit": "Unit Testing",
        "GoogleTest": "Unit Testing",
        
        # OOP variations
        "Object-Oriented": "OOP",
        "object-oriented": "OOP",
        
        # npm variations
        "NPM": "npm",
        
        # Data structure variations
        "DSA": "Data Structure",
        "data structure": "Data Structure",
        "complexity analysis": "Data Structure",
        
        "operating systems": "OS",
        "Operating Systems": "OS",
        
        "Linux/Unix": "Linux",
        
        "Rabbit": "RabbitMQ",
        
        "JAVA": "Java",
        
        "Power BI": "PowerBI",
        
        "PowerShell": "Shell",
        
        "Postgres": "PostgreSQL",
        
        "DEVOPS": "DevOps",
        
        "computer vision algorithms": "Computer Vision Algorithms",
        
        "networking protocols": "Networking Protocols",
        
        "multithreaded": "Multithreading",
        "multi-threading": "Multithreading",
        
        "JPA": "JPA2",
        "DB2": "Db2",
        
        "microservices": "Microservices",
        "Micro-Services": "Microservices",
        "Micro-services": "Microservices",
        
        "optical fibers": "Optical Fibers",
        
        "JIRA": "Jira",
        
        "S3": "Amazon S3",
        "Amazon Web Services": "AWS",
        
        "flux": "Flux",
        
        "RushJs": "Rush",
        "RushJS": "Rush",
        "Rush.js": "Rush",
        
        "GCP": "Google Cloud",
        "Google Cloud Platform": "Google Cloud",
        
        "bootstrap": "Bootstrap",
        "regression": "Regression",
        
        "Express.js": "Express",
        
        "blockchain": "Blockchain",
        
        "linear programming": "Linear Programming",
        
        "multi-tiered systems": "Multi-tiered Systems",
        
        "Rx.js": "RxJS",
        "RxJS": "RxJS",
        "rxjs": "RxJS",
        
        "relational databases": "Relational Databases",
        "relational database": "Relational Databases",
        "Relational Database": "Relational Databases"
    }

def create_compiled_patterns(possible_skills, special_regex_skills):
    """Precompile regex patterns for better performance."""
    compiled_patterns = {}
    for skill in possible_skills:
        if skill in special_regex_skills:
            pattern = special_regex_skills[skill]
        else:
            # Basic word boundary for most skills
            if re.match(r'^[a-zA-Z]', skill) and not re.search(r'\W', skill):
                pattern = r'\b' + re.escape(skill) + r'\b'
            else:
                pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
        compiled_patterns[skill] = re.compile(pattern, re.IGNORECASE)
    return compiled_patterns

def count_skills(text, compiled_patterns, skill_mapping):
    """Count skill occurrences using compiled patterns."""
    skill_counts = Counter()
    
    # First, count direct matches
    for skill, pattern in compiled_patterns.items():
        count = len(pattern.findall(text))
        if count > 0:
            skill_counts[skill] = count
    
    # Then, handle aliases in skill_mapping
    for alias, normalized in skill_mapping.items():
        # If the value is a list, it means multiple normalized forms
        if isinstance(normalized, list):
            normalized_skills = normalized
        else:
            normalized_skills = [normalized]
            
        # Create pattern for the alias
        if re.match(r'^[a-zA-Z]', alias) and not re.search(r'\W', alias):
            alias_pattern = r'\b' + re.escape(alias) + r'\b'
        else:
            alias_pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
        
        alias_count = len(re.findall(alias_pattern, text, re.IGNORECASE))
        
        # Add counts to normalized skill(s)
        if alias_count > 0:
            for norm_skill in normalized_skills:
                if norm_skill in compiled_patterns:
                    skill_counts[norm_skill] += alias_count
    
    return skill_counts

def process_batch(pdf_batch, compiled_patterns, skill_mapping):
    """Process a batch of PDFs: remove duplicates per PDF, combine, then count skills."""
    batch_text = ""
    for pdf in pdf_batch:
        text = extract_pdf_text(pdf)
        # Remove duplicate words in this PDF before adding it to the batch
        unique_text = remove_duplicates_in_text(text)
        batch_text += unique_text + " "
    
    return count_skills(batch_text, compiled_patterns, skill_mapping)

def main():
    args = parse_arguments()
    
    # Get list of PDF files from all directories
    pdf_files = []
    for pdf_dir in args.pdf_dirs:
        if not os.path.exists(pdf_dir):
            logging.error(f"Directory not found: {pdf_dir}")
            print(f"Error: Directory not found: {pdf_dir}")
            continue
            
        dir_files = glob.glob(f"{pdf_dir}/*.pdf")
        if len(dir_files) == 0:
            logging.error(f"No PDF files found in {pdf_dir}")
            print(f"Error: No PDF files found in {pdf_dir}")
            continue
            
        pdf_files.extend(dir_files)
    
    total_files = len(pdf_files)
    
    if total_files == 0:
        logging.error("No PDF files found in any directory")
        print("Error: No PDF files found in any directory")
        return
    
    print(f"Found {total_files} PDF files to analyze")
    logging.info(f"Found {total_files} PDF files across all directories")
    
    # Define possible skills
    possible_skills = [
        "C++","C","C#","Python","Pandas","AWS","Java","JavaScript","Go","SQL","REST","TypeScript",
        "React","Node","API","Excel","Linux","Agile","Cloud","ML","Oracle","Scrum","Jira","Windows",
        "Docker",".NET","Word","Shell","Ruby","Perl","Kubernetes","Unix","Selenium","Visual Studio",
        "VS Code","Swift","Scala","Hadoop","Git","GitHub","Tableau","PowerBI","Microsoft Office",
        "Jenkins","TensorFlow","Matlab","CI/CD","Django","Google Cloud","Robot Framework",
        "Cloud Computing","Azure","Jupyter","Salesforce","Keras","Kafka","PyTorch","Figma","Angular",
        "Assembly","Arduino","Unreal Engine","Big Data","Rust","Flask","Vue","Matplotlib",
        "Objective-C","Anaconda","Redis","CLI","Vercel","BitBucket","npm","PNPM","Remix","Tanstack",
        "HTML","CSS","ShadCN","RadixUI","MUI","TCP/IP","gRPC","HTTP","TLS","Web Sockets",
        "GitHub Actions","Springboot","DevOps","SDLC","VBA","Pyautogui","Razor","Xamarin","Maui",
        "Controller Area Network","ISOBUS","RS232","Unit Testing","JSON","Simulink","LiDAR",
        "Apache Spark","Apache Hop","Terraform","Databricks","Unity","Object-Oriented",
        "Data Structure","XML","XSD","SOAP","Eclipse","SuiteBuilder","NetSuite","SuiteBundler",
        "SuiteFlow","SuiteTalk","Celigo","Integrator.io","Dell Boomi","SPS","Spring Boot","Kotlin",
        "R","AutoCAD","SharePoint","Dojo","Groovy","Artifactory","IBM Cloud","NoSQL","MySQL","Db2",
        "Cloudant","OpenShift","Ansible","Cypress","ngrx","RxJS","GraphQL","Apollo","RTOS","Ethernet",
        "SPI","I2C","ARM","ATOM","DSP","FPGA","GPU","LIN","UDS","CANApe","UIKit","LLMs",
        "Hugging Face","OpenCV","AI","SDK","Xcode","Android Studio","PostgreSQL","MongoDB",
        "Rails","XSL","Perforce","OAuth","SSL","NDK","JNI","Jetpack","DQN","LSTM","SVM",
        "random forest","VFX","OpenGL","WebGL","DirectX","Metal","Vulkan","Verilog","SystemVerilog",
        "VHDL","Digital Signal Processing","bash","Tcl","LangChain","Redux","Helm","LabVIEW",
        "JQuery","GitLab","Julia","NLP","scikit-learn","sklearn","GUI","DAST","SAST","SCA","CVE",
        "CVSS","CWE","ETL","PowerPoint","Kanban","Playwright","Postman","TeamCity","OpenAI",
        "Sagemaker","Metaflow","YAML","SCM","JTAG","PCIe","CXL","FME Flow Hosted","FME Hub",
        "FME Account","FME AI Service","Spring","FastAPI","Apache Tomcat","JBoss","Websocket",
        "JPA2","ORM","Hibernate","Cassandra","HBase","Flink","cmake","5G","TestCafe",
        "Computer Vision Algorithms","Networking Protocols","Multithreading","Outlook","Gerrit",
        "S3","Microservices","T-SQL","Knockout","JSCON","OData","ERD","SSRS","Flutter","Express",
        "GenAI","Bamboo","Apex","CRM","SaaS","SAP Analytics","DNS","Endpoint Protection",
        "Web Proxy","Vulnerability Management","SIEM","Access","BigQuery","Compute Engine","XUnit",
        "PHP","IoT","Raspberry Pi","Pub/Sub","asyncio","VMWare","QEMU","Security Operations",
        "Threat Containment","Zero Trust","SVN","MS Build","Optical Fibers","Open RAN","Elixir",
        "RabbitMQ","LAN","WAN","MS Visio","D3","Vite","Rush","Flux","ECMAScript","K8S","Ember",
        "CCNA","Bootstrap","IQR","K-S test","Regression","ARIMA","DHCP","VLANS","OSPF","BGP",
        "ESXi","Xen","InfluxDB","Multi-tiered Systems","Integration Testing","Relational Databases",
        "Linear Programming","Electron","UDP","OpenStack","PKI","Vagrant","Gradle","RF Test","ARP",
        "Packer","Cloud-Init","Kickstart","Blockchain","JAAS","SuiteScript","React Native","OpenCL",
        "Direct3D","UML","Trello","Adobe Creative","Metabase","Next.js","Comment Standard",
        "Argo Workflows","particle.js","ROS","Gazebo","BuildBot","Solidity","TailwindCSS","SMACSS",
        "Tauri","Astro","SvelteKit","Qwik","MobX","SolidJS","SAML","Kerberos","SSO","ABAP",
        "SAP HANA","Nuxt 3","Firebase","Firestore","Web3","Gatsby","VIM","ThreeJS","GLSL","JAMStack",
        "Elm","Webpack","LevelDB", "Discord.js", "Star Schema", "Waterfall", "Hive", "Beeline", "HDFS",
        "Sqoop", "MinIO", "Airflow", "Trino", "PyHive", "PySpark", "Korn"
    ]
    
    special_regex_skills = {
        "C": r"\bC(?!\+)\b|\bC programming\b|\bC language\b|\bC developer\b",
        "R": r"\bR\b|\bR programming\b|\bR language\b|\bR developer\fb",
        # Other abbreviations that need specific matching
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
        "Helm": r"\bHelm\b"
    }
    
    # Create skill mapping and compiled patterns
    skill_mapping = create_skill_mapping()
    compiled_patterns = create_compiled_patterns(possible_skills, special_regex_skills)
    
    # Prepare for batching
    batch_size = args.batch_size
    batches = [pdf_files[i:i+batch_size] for i in range(0, len(pdf_files), batch_size)]
    print(f"Processing {len(batches)} batches with batch size {batch_size}")
    
    all_skill_counts = Counter()
    
    # Process batches with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(batches))) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, compiled_patterns, skill_mapping): batch
            for batch in batches
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                           total=len(batches), desc="Processing PDF batches"):
            try:
                batch_counts = future.result()
                all_skill_counts.update(batch_counts)
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
    
    # Sort results and display
    sorted_skills = sorted(all_skill_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nSkill Counts:")
    for skill, count in sorted_skills:
        print(f"{skill}: {count}")
    
    # Save to CSV
    results_df = pd.DataFrame(sorted_skills, columns=['Skill', 'Count'])
    output_path = os.path.join(os.path.dirname(args.pdf_dirs[0]), 'skill_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()