Learnfiy – AI Learning platform
Syllabus-aligned digital learning support for Zimbabwe’s  A ‘Level Commercial learners
One-line overview
PassPlus Learning helps students keep up with the syllabus and prepare for exams—without relying on expensive extra lessons- by turning learning data into clear, personalized “what to do next” study recommendations.
The Problem
Across Zimbabwe, many families are spending more on education because learners often depend on paid extra lessons and holiday lessons in addition to school fees. In practice, extra lessons can start to feel less like an option and more like a requirement for staying on track and doing well in examinations.
This pressure is made worse when learning time at school is disrupted (for example, during go-slow action or strikes). Less teaching time can mean the syllabus isn’t fully covered and revision is rushed. When extra lessons also become a personal income source, families may feel that the real support only comes in paid sessions.
The impact is both financial and emotional. Costs add up quickly for learners taking many subjects, and students who can’t afford extra lessons may feel left behind, lose confidence, and participate less. Over time, performance risks becoming tied to ability to pay rather than effort and ability.
The Solution
Learnfiy Learning is a syllabus-aligned study support platform that helps students learn outside the classroom and prepare for exams at a lower cost.
It offers topic-by-topic lessons, quizzes, AI help (text/voice), and PDF/Word upload marking with clear feedback. Parents/guardians can monitor progress for free once a learner is subscribed. For now, it focuses on A’Level commercial subjects: Accounting, Business Studies, Mathematics, Economics, and Computer Science. It will be free for the next 3 months as a trial while we test and improve it. After that, pricing (if learners find it helpful) will be Primary $5/month, O’Level $10/month, A’Level $15/month (all subjects per level). This is far cheaper than paying per subject (e.g., 10 subjects can cost around $200/month with extra lessons). Internet is needed for now, but an offline-friendly version is planned for future updates.
Prescriptive Analytics Approach (Data → Model → Recommendation)
Learnfiy is designed to be prescriptive. It doesn’t only report scores; it recommends the next best actions to improve outcomes while respecting real constraints like limited study time, limited data/connectivity, and limited time before exams.
1) Data inputs
Our platform is an AI-powered learning system for commercial students. After logging in and selecting a subject (e.g., Accounting), students can use an AI Tutor to ask questions or upload documents for instant support. The platform also enables homework uploads and submission tracking, subject-based quizzes, and performance analytics to measure student progress over time.”
2) Models and logic
Learnfiy uses a simple, explainable pipeline that can grow in sophistication over time:
• Topic mastery estimation: rules-based mastery scoring (MVP), with an upgrade path to Bayesian Knowledge Tracing (BKT) or IRT-style scoring.
• Risk/readiness prediction: interpretable models (e.g., logistic regression) that flag “at-risk” learners using mastery gaps + engagement trends.
• Prescriptive planning: a constrained study plan generator (heuristics + spaced repetition). It recommends topics, activities, and timing that fit the learner’s available minutes and time-to-exam.
3) Recommendation outputs
Learners  receive actionable outputs such as:
• Next best topic to study (and why)
• A realistic weekly plan (sessions, topics, and suggested homework depending your weak areas)
• Targeted practice sets based on error patterns
• Revision reminders using spaced repetition
• Parent-friendly alerts when a learner is falling behind or showing high risk
Live Demo
Try it here: (add your Streamlit URL)
Screenshot: (add screenshot.png in your repo / app page)
How It Works
•	Login and Subject Selection -Students log in and select a subject from the available subject list (e.g., Accounting).
•	AI Tutor (Q&A and Document Help): Inside the subject, students can: -Ask the AI Tutor any question using a search bar. Upload a document (e.g., notes or a worksheet) and ask questions based on the content uploaded.  The system returns relevant answers automatically, and results appear directly in the interface.
•	Homework Support and Submission - Students can upload homework tasks through the Homework section.  Once uploaded, the system shows the homework status (e.g., submitted) and keeps a record of the submission.
•	Quizzes and Learning Assessment - After studying, students can take quizzes within the selected subject to test understanding and strengthen key concepts.
•	 Analysis and Progress Tracking - The platform provides analytics that show how the student is progressing over time. A Progress section measures performance and helps monitor improvement during the learning
Level-Specific Examples
A’Level 
At A’Level, deeper understanding and exam technique matter. Upload marking provides strong signals for gaps (method selection, explanation structure, and time management). If a learner struggles with integration techniques, Learnfiy prescribes a sequence: focused lesson, worked examples, targeted practice, and a timed mini-task. The AI assistant provides step-by-step help on demand, and the platform checks improvement by comparing performance before and after the intervention.
Technology Stack (typical implementation)
• Frontend: Streamlit
• ML: scikit-learn (mastery scoring MVP and risk prediction)
• Planning/Optimization: heuristics + spaced repetition (optional: Gurobi or SciPy for constrained scheduling)
• Data: learner-generated activity data (lessons, quizzes, marking outcomes, engagement)
About This Project
Built for ISOM 839 (Prescriptive Analytics) at Suffolk University.
Author: GILBERT ZINYENYA
LinkedIn: www.linkedin.com/in/gilbert-zinyenya-263b9418b
Email: gzinyenya@gmail.com
Future Possibilities
• Upgrade mastery estimation from rules-based scoring to BKT/IRT for more robust probabilities.
• Add A/B testing to validate impact of prescriptive recommendations versus generic sequencing.
• Improve low-data / offline-friendly delivery for learners with limited connectivity.
• Expand parent dashboard with simple weekly guidance and clearer “at-risk” explanations.
• Add school-level reporting (optional) to complement classroom learning without replacing teachers.
Demo Video
Watch the walkthrough: (https://suffolk.zoom.us/rec/share/gP7LQ-ko0rFSFIldeZDnxT7Apo2koDYQ6fJhssZ1Cm21TWx1r9Vp7u9U3FUhvq8x.45mJ3sqFIii-6c2M)


