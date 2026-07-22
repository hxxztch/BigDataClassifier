path = "E:/Study/Spark大数据快速分类/BigDataClassifier/frontend/index.html"
with open(path, "r", encoding="utf-8") as f:
    html = f.read()

# Remove the wrongly placed data quality button HTML from data section
# It's between sceneDetailDialog and dqLoading
old = """                    sceneDetailDialog: false,
                                    <el-button type="primary" icon="el-icon-document-checked" @click="checkDataQuality" :loading="dqLoading" style="margin-right: 12px;">
                                     {{ dqLoading ? "质检中..." : "数据质检" }}"""

new = """                    sceneDetailDialog: false,"""

html = html.replace(old, new)

# Also fix similar issue with the _this.dqLoading line that got misplaced
old2 = """                                    _this.dqLoading = true;
                                    _this.dqResult = null;"""

# Remove from wherever it is (it should be in checkDataQuality method)
if old2 in html:
    html = html.replace(old2, "")
    print("Removed misplaced dqLoading line")

with open(path, "w", encoding="utf-8") as f:
    f.write(html)

import shutil
shutil.copy2(path, "E:/Study/Spark大数据快速分类/BigDataClassifier/code/Spark_New/frontend/index.html")

# Verify
lines = open(path, "r", encoding="utf-8").readlines()
for i in range(537, min(548, len(lines))):
    print(f"L{i+1}: {lines[i].rstrip()[:80]}")

print("Fixed")
