# Streamlit Cloud Deployment Troubleshooting

## Current Issue
Getting "installer returned a non-zero exit code" error during deployment.

## Solutions to Try

### 1. Check the Actual Error Logs
In Streamlit Cloud dashboard, click on "Manage App" and check the terminal/logs to see the specific package causing the failure.

### 2. Python Version
Currently set to Python 3.11 in `runtime.txt`. If issues persist, try:
- Python 3.10: `python-3.10`
- Python 3.9: `python-3.9`

### 3. If cvxpy/scikit-learn Build Fails
These packages require compilation. Options:
- Wait longer (first build can take 10-15 minutes)
- Use pre-built wheels by ensuring compatible versions
- Consider using alternative packages if deployment continues to fail

### 4. Manual Dependency Installation
If automatic installation fails, you may need to:
- Install build dependencies first
- Use a different package manager
- Contact Streamlit support

### 5. Alternative: Deploy Without Heavy Dependencies
If deployment continues to fail, consider:
- Creating a lighter version without DoWhy/EconML for demo
- Using local deployment instead
- Using a different hosting platform (Heroku, Railway, etc.)

## Next Steps
1. Check the detailed error logs in Streamlit Cloud
2. Share the specific error message for targeted fix
3. Consider if a simplified version would work for demonstration

