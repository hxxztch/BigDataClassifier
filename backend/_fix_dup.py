path = "E:/Study/Spark大数据快速分类/BigDataClassifier/frontend/index.html"
with open(path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Fix corrupted Color: -> borderColor:
html = html.replace("rgba(38,68,106,0.9),Color: #64ffda", "rgba(38,68,106,0.9), borderColor: #64ffda")
html = html.replace("rgba(23,42,69,0.9),Color: #64ffda", "rgba(23,42,69,0.9), borderColor: #64ffda")

# 2. Find and remove duplicate renderCharts
# First renderCharts
first_start = html.find("renderCharts() {")
first_end = html.find("                    },", first_start)

# Second renderCharts (after first end)
second_start = html.find("renderCharts() {", first_end)
second_end = html.find("                    },", second_start)

print(f"First: {first_start}->{first_end}, Second: {second_start}->{second_end}")

# Remove second renderCharts (from its opening to its closing ),)
# Find the actual end: the }, that closes the method
second_close = html.find("\n                    },", second_start)
# Remove from second_start to after the closing
html = html[:second_start] + html[second_close+1:]

# 3. Also fix `fixBorders` remnants if any
html = html.replace("function fixBorders()", "/* fixBorders removed")

with open(path, "w", encoding="utf-8") as f:
    f.write(html)

import shutil
shutil.copy2(path, "E:/Study/Spark大数据快速分类/BigDataClassifier/code/Spark_New/frontend/index.html")
print("Fixed: removed duplicate renderCharts, restored borderColor")
