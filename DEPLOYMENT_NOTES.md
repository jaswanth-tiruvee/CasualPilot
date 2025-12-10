# Deployment Notes

## Current Status
Experiencing build errors on Streamlit Cloud. The error "installer returned a non-zero exit code" suggests a package installation failure.

## To Diagnose:
1. Check Streamlit Cloud logs for the specific error
2. Look for which package fails (likely cvxpy, scikit-learn, or their dependencies)
3. Check if it's a build error or dependency conflict

## Potential Solutions:

### Option 1: Use Pre-built Wheels
Ensure all packages can install from pre-built wheels (no compilation needed)

### Option 2: Add Build Dependencies
May need to add build tools in packages.txt (though Streamlit Cloud usually has these)

### Option 3: Alternative Deployment
- Use Railway.app (better for complex dependencies)
- Use Heroku (more control over build process)
- Use local deployment with ngrok for demo

### Option 4: Simplify Dependencies
Create a demo version without DoWhy/EconML that shows the concept

## Next Steps:
1. Get the actual error message from logs
2. Identify the failing package
3. Apply targeted fix

