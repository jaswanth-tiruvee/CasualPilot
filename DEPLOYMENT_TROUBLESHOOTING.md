# Streamlit Cloud Deployment Troubleshooting

## Quick Fixes for Common Errors

### Error: "ModuleNotFoundError: No module named 'X'"
**Solution**: Ensure all dependencies are in `requirements.txt` and pushed to GitHub.

### Error: "Build failed" or "Installation timeout"
**Possible causes**:
- Conflicting package versions
- Large dependencies (EconML/DoWhy take time to install)

**Solutions**:
1. Check `requirements.txt` for version conflicts
2. Wait - first deployment can take 5-10 minutes
3. Try pinning specific versions if conflicts persist

### Error: "MemoryError" or "Out of memory"
**Solution**: Reduce default sample size in `app.py`:
```python
n_samples = st.sidebar.slider("Number of Samples", 1000, 5000, 3000, 500)  # Reduced max
```

### Error: "AttributeError" or "KeyError"
**Solution**: 
- Clear browser cache
- Ensure session state is properly initialized
- Check that data columns match expected names

### Error: "Import error" for local modules
**Solution**: Ensure all Python files are in the root directory:
- `app.py`
- `causal_pipeline.py`
- `data_generator.py`
- `visualization.py`
- `targeting_policy.py`

## Deployment Checklist

- [ ] All files pushed to GitHub
- [ ] `requirements.txt` is in root directory
- [ ] `app.py` is in root directory
- [ ] No syntax errors (tested locally)
- [ ] All imports work locally
- [ ] Streamlit Cloud connected to correct GitHub repo
- [ ] Main file path set to `app.py`

## Testing Locally Before Deployment

```bash
# Test imports
python -c "import streamlit; import dowhy; import econml; print('OK')"

# Test app runs
streamlit run app.py

# Check syntax
python -m py_compile app.py
```

## If Deployment Still Fails

1. Check Streamlit Cloud logs (most detailed error info)
2. Share the exact error message
3. Verify Python version compatibility (3.8+)
4. Try deploying with minimal sample size first

