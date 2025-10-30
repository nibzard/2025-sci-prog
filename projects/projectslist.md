# Pair Programming Projects

This document contains instructions and guidelines for completing projects in pairs for the Scientific Programming course. All projects are completed collaboratively using modern GitHub workflows.

## 🎯 Learning Objectives

- Practice pair programming and collaborative development
- Learn professional GitHub workflows (issues, pull requests, code review)
- Develop communication and teamwork skills
- Create meaningful scientific programming applications

## 🚀 Getting Started

### Step 1: Form Your Team
1. Find a partner to work with (teams of 2 students)
2. Choose a creative team name
3. Decide on a project idea that interests both of you

### Step 2: Set Up Your Project Folder
Create your project folder using this naming convention:
```
team_name-student1-student2/
```

**Examples:**
- `climate-analyzers-jdoe-msmith/`
- `data-wizards-abrown-cjohnson/`
- `python-gurus-dlee-jchen/`

### Step 3: GitHub Repository Setup
1. **Fork this repository** on GitHub
2. **Clone your fork** to your local machine:
   ```bash
   git clone [your-fork-url]
   cd 2025-sci-prog
   ```
3. **Create your project folder** in the `projects/` directory
4. **Create a GitHub Issue** for your project planning

## 🔄 GitHub Collaboration Workflow

### Branch Strategy
- `main`: Protected branch for final code
- `develop`: Integration branch for work in progress
- `feature/[feature-name]`: Individual feature branches

### Daily Workflow

1. **Start of Day**
   - Sync with partner: Discuss goals and who's driving/navigating
   - Pull latest changes: `git pull origin main`
   - Create feature branch: `git checkout -b feature/[feature-name]`

2. **During Development**
   - **Driver**: Writes the code
   - **Navigator**: Reviews code, thinks ahead, catches issues
   - **Switch roles** every 25-30 minutes (Pomodoro technique)
   - **Commit frequently** with descriptive messages

3. **End of Day**
   - **Push your branch**: `git push origin feature/[feature-name]`
   - **Create Pull Request** to main branch
   - **Partner reviews** the PR (must be approved by both team members)
   - **Merge to main** after both approve

### Pull Request Template
```markdown
## What We Did
- [ ] Feature 1 implemented
- [ ] Feature 2 implemented
- [ ] Tests added and passing

## How to Test
1. Run `python main.py`
2. Expected output: ...

## What We Learned
- Brief reflection on what you learned today

## Next Steps
- What you plan to work on next
```

## 📋 Project Submission Format

Add your project to the list below using this format:

```markdown
### [Team Name] - [Student 1] & [Student 2]
* **Project:** [Project Title]
* **Description:** [One-line description]
* **GitHub Issues:** [Link to your project planning issue]
* **Technologies:** [List of main technologies used]
```

---

## Current Projects

*(Teams will add their projects here as they get started)*

## 🛠 Pair Programming Best Practices

### Communication
- **Think aloud** when you're the driver
- **Ask questions** when you're the navigator
- **Take breaks** together every hour
- **Be patient** and respectful of different coding styles

### Code Quality
- **Test your code** before committing
- **Write clear commit messages**
- **Review each other's code** carefully
- **Refactor together** when needed

### Problem Solving
- **Google together** - don't waste time stuck
- **Draw diagrams** on paper or virtual whiteboard
- **Explain concepts** to each other
- **Ask for help** from instructors when needed

## 📊 Project Requirements

### Technical Requirements
- [ ] Use Python (or language approved by instructor)
- [ ] Include proper error handling
- [ ] Add comments and documentation
- [ ] Create a README.md for your project
- [ ] Use Git appropriately (commits, branches, PRs)

### Collaboration Requirements
- [ ] Both team members contribute code
- [ ] Use GitHub Issues for project planning
- [ ] Create Pull Requests for all features
- [ ] Both members must approve PRs
- [ ] Document your learning process

## 🎉 Project Showcase

At the end of the course, teams will:
1. **Present their project** to the class
2. **Demonstrate the application**
3. **Share collaboration experiences**
4. **Discuss challenges and solutions**

## ❓ Need Help?

- **Create a GitHub Issue** in the main repository
- **Ask during class** or office hours
- **Use Discord/Slack** for quick questions
- **Review GitHub documentation** for workflow questions

---

**Remember**: The goal is to learn both technical skills AND collaboration skills. Communication and teamwork are just as important as the code you write!