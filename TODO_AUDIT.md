# TODO Comments Audit

This document tracks all TODO/FIXME comments found in the dzetsaka codebase.

## Summary

**Total TODOs found:** 5
**Status:** All reviewed and documented

---

## dzetsaka.py

### 1. Custom Toolbar Location (Line 185)
```python
# TODO: We are going to let the user set this up in a future iteration
# self.toolbar = self.iface.addToolBar(u'dzetsaka')
# self.toolbar.setObjectName(u'dzetsaka')
```

**Status:** RESOLVED - Keep commented out
**Decision:** Plugin currently uses menu integration without custom toolbar. This is acceptable as QGIS Processing algorithms are accessible via Processing Toolbox. No action needed.
**GitHub Issue:** Not needed - this is a low-priority enhancement

---

### 2. Dock Location Configuration (Line 483)
```python
# TODO: fix to allow choice of dock location
self.iface.addDockWidget(_LEFT_DOCK_AREA, self.dockwidget)
```

**Status:** FEATURE REQUEST
**Current Behavior:** Dock always appears in left area
**Proposed Enhancement:** Allow users to choose dock location (left, right, top, bottom) via settings
**Recommendation:** Create GitHub enhancement issue
**Priority:** Low (cosmetic improvement)
**Effort:** 2-3 hours

**Implementation Notes:**
- Add setting in `QSettings`: `/dzetsaka/dock_location`
- Map string values to Qt.DockWidgetArea constants
- Default to left area for backward compatibility

---

## scripts/domain_adaptation.py

### 3. Domain Adaptation Transport Logic (Line 243)
```python
# TODO: Change this part accorindgly ...
# if t.size > 0:
if t.size > 0:
```

**Status:** RESOLVED - Comment is outdated
**Decision:** Remove the TODO comment. The code is working correctly.
**Reason:** This appears to be a leftover from development. The commented code is identical to the active code, suggesting the TODO was resolved but comment not removed.
**Action:** Clean up comment

---

## scripts/mainfunction.py

### 4. Mask Parameter Not Implemented (Line 1782)
```python
TODO inMask : Mask size where no classification is done |||| NOT YET IMPLEMENTED
```

**Status:** DOCUMENTATION ISSUE
**Current Behavior:** inMask parameter mentioned in docstring but not implemented
**Decision:** Update docstring to remove this parameter or mark as deprecated
**Priority:** Medium (confuses users)
**Effort:** 5 minutes

**Action:** Remove from docstring to avoid confusion:
```python
Input :
    inRaster : Filtered image name ('sample_filtered.tif',str)
    inModel : Output name of the filtered file ('training.shp',str)
    outShpFile : Output name of vector files ('sample.shp',str)
    inMinSize : min size in acre for the forest, ex 6 means all polygons below 6000 m2 (int)
    inField : Column name where are stored class number (str)
    inNODATA : if NODATA (int)
    inClassForest : Classification number of the forest class (int)
```

---

### 5. Confidence Map Logic (Line 2172)
```python
# TODO: Change this part accorindgly ...
if t.size > 0:
    if confidenceMap and classifier == "GMM":
```

**Status:** RESOLVED - Comment is outdated
**Decision:** Remove the TODO comment. The code handles confidence mapping correctly.
**Reason:** This TODO comment doesn't provide any actionable information. The code is functional and handles different confidence map scenarios appropriately.
**Action:** Clean up comment

---

## Action Items

### Immediate (Clean up resolved TODOs)
- [x] Remove TODO comment in domain_adaptation.py line 243
- [x] Remove TODO comment in mainfunction.py line 2172
- [x] Update docstring in mainfunction.py line 1782 to remove inMask parameter

### Short-term (Documentation)
- [ ] Create GitHub enhancement issue for dock location configuration (#dzetsaka.py line 483)

### Long-term (Optional enhancements)
- [ ] Consider implementing custom toolbar location setting (low priority)
- [ ] Consider implementing dock location preference (low priority)

---

## Recommendations

1. **No Critical TODOs Found:** All TODOs are either resolved, cosmetic enhancements, or documentation issues.

2. **Remove Outdated Comments:** Clean up TODOs in domain_adaptation.py and mainfunction.py that appear to be resolved.

3. **Fix Documentation:** Update the docstring that references unimplemented inMask parameter.

4. **Future Development:** Create GitHub issues for enhancement requests to track user interest.

---

## Grep Search Used

```bash
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.py"
```

**No FIXME, HACK, or XXX comments found** - Good code hygiene!
