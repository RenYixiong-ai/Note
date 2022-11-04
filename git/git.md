# Git简单教程

工作逻辑图：
![逻辑图](./workflow.png)

reference:
* [莫凡PYTHON]("https://mofanpy.com/tutorials/others/git")

# 基础操作

创建一个新的仓库

```
    git init
```

查看当前版本库状态

```
    git status
```

增加进版本库

```
    git add .       #添加所有更改进版本库
    git add 文件     #添加具体文件进版本库
```

提交改变

```
    git commit -m "这次提交的备注"
```

# 回到从前

## 查看历史版本

```
    git lod --oneline
```

## 回到过去的版本

```
# 不管我们之前有没有做了一些 add 工作, 这一步让我们回到 上一次的 commit
$ git reset --hard HEAD    
# 输出
HEAD is now at 904e1ba change 2
-----------------------
# 看看所有的log
$ git log --oneline
# 输出
904e1ba change 2
c6762a1 change 1
13be9a7 create 1.py
-----------------------
# 回到 c6762a1 change 1
# 方式1: "HEAD^"
$ git reset --hard HEAD^  

# 方式2: "commit id"
$ git reset --hard c6762a1
-----------------------
# 看看现在的 log
$ git log --oneline
# 输出
c6762a1 change 1
13be9a7 create 1.py
```

## 在回到历史版本之后再回来

```
#第一步
$ git reflog
# 输出
c6762a1 HEAD@{0}: reset: moving to c6762a1
904e1ba HEAD@{1}: commit (amend): change 2
0107760 HEAD@{2}: commit: change 2
c6762a1 HEAD@{3}: commit: change 1
13be9a7 HEAD@{4}: commit (initial): create 1.py

#第二步
$ git reset --hard 904e1ba
$ git log --oneline
# 输出
904e1ba change 2
c6762a1 change 1
13be9a7 create 1.py
```

## 单个文件回到过去

利用 ```checkout``` 将过去版本的文件覆盖现在同名文件，然后作为当前版本的一个新的更新，更新进版本库。

```
$ git log --oneline
# 输出
904e1ba change 2
c6762a1 change 1
13be9a7 create 1.py
---------------------
$ git checkout c6762a1 -- 1.py  #checkout + id + -- + 文件名字
```

然后增加进版本库并提交改变

```
$ git add 1.py
$ git commit -m "back to change 1 and add comment for 1.py"
$ git log --oneline

# 输出
47f167e back to change 1 and add comment for 1.py
904e1ba change 2
c6762a1 change 1
13be9a7 create 1.py
```