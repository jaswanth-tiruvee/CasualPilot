# Streamlit Cloud Deployment Guide

## Common Issues and Solutions

### 1. Import Errors
If you see import errors, ensure all dependencies are in `requirements.txt`:
- DoWhy
- EconML
- All other dependencies

### 2. Memory Issues
EconML and DoWhy can be memory-intensive. If deployment fails:
- Reduce default sample size in the app
- Use lighter HTE methods (DML instead of Causal Forest)

### 3. Build Timeout
If the build times out:
- Check that requirements.txt doesn't have conflicting versions
- Ensure all dependencies are compatible

### 4. Runtime Errors
Common runtime errors:
- **AttributeError**: Check that session state is properly initialized
- **KeyError**: Ensure data columns match expected names
- **MemoryError**: Reduce sample size

## Deployment Steps

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Deploy!

## Configuration

The app uses:
- Main file: `app.py`
- Python version: 3.8+ (auto-detected)
- Dependencies: `requirements.txt`

## Troubleshooting

If deployment fails:
1. Check the logs in Streamlit Cloud dashboard
2. Verify all imports work locally: `python -c "import streamlit; import dowhy; import econml"`
3. Test locally: `streamlit run app.py`
4. Check requirements.txt for version conflicts

