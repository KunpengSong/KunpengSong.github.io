# Python Cheat Sheet

## Numpy

1. arrange(start, end, step): return a numpy array in range [start, end, stepwidth)

```python
>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])
```

2. reshape((dim,dim)): return reshaped array ,-1 means auto calculated.  

3. Find the most frequent number in a numpy vector

   ```python
   a = np.array([1,2,3,1,2,1,1,1,3,2,2,1])
   np.bincount(a).argmax()
   ```

4. Scale Numpy array to certain range

   NumPy provides numpy.interp for 1-dimensional linear interpolation. In this case, where you want to map the minimum element of the array to −1 and the maximum to +1, and other elements linearly in-between, you can write:

   ```python
   np.interp(a, (a.min(), a.max()), (-1, 1))
   ```

5. numpy.random.permutation

   ```python
   >>> np.random.permutation(10)
   array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
   >>> np.random.permutation([1, 4, 9, 12, 15])
   array([15,  1,  9,  4, 12])
   >>> arr = np.arange(9).reshape((3, 3))
   >>> np.random.permutation(arr)
   array([[6, 7, 8],
          [0, 1, 2],
          [3, 4, 5]])
   ```

6. copy a file in Python

   ```python
   import shutil
   shutil.copy2('/src/dir/file.ext', '/dst/dir/newname.ext') # complete target filename given
   shutil.copy2('/src/file.ext', '/dst/dir') # target filename is /dst/dir/file.ext
   ```

   

## OS and File system

## Others:

1. enumerate(sequence, 0) : sequence: 序列、迭代器或其他支持迭代对象。 start: 标起始位置。返回枚举对象

   ```python
   >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
   >>> list(enumerate(seasons))
   [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
   >>> list(enumerate(seasons, start=1))       # 下标从 1 开始
   [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
   
   >>>seq = ['one', 'two', 'three']
   >>> for i, element in enumerate(seq):
   ...     print i, element
   0 one
   1 two
   2 three
   ```

2. dict.items() : Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。

   ```python
   dict = {'Google': 'www.google.com', 'Runoob': 'www.runoob.com'}
   # 遍历字典列表
   for key,values in  dict.items():
       print(key,values)
   ```

3. random.seed([x] ) : x -- 改变随机数生成器的种子seed

   ```python
   >>> random.seed( 10 )
   >>> print "Random number with seed 10 : ", random.random()
   # 生成同一个随机数
   >>> random.seed( 10 )
   >>> print "Random number with seed 10 : ", random.random()
   Random number with seed 10 :  0.57140259469
   Random number with seed 10 :  0.57140259469
   ```

4. np.rot90(array,times=1,axis=None) : 

   ```python
   >>> m = np.array([[1,2],[3,4]], int)
   >>> m
   array([[1, 2],
          [3, 4]])
   >>> np.rot90(m)
   array([[2, 4],
          [1, 3]])
   >>> np.rot90(m, 2)
   array([[4, 3],
          [2, 1]])
   >>> m = np.arange(8).reshape((2,2,2))
   >>> np.rot90(m, 1, (1,2))
   array([[[1, 3],
           [0, 2]],
          [[5, 7],
           [4, 6]]])
   ```

5. reshape(): 

   ```python
   a = np.array(range(0,24))
   print(a)
   print(a.reshape(3,2,4)) # actual permutation: 4,2,3
   [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
   [[[ 0  1  2  3]
     [ 4  5  6  7]]
    [[ 8  9 10 11]
     [12 13 14 15]]
    [[16 17 18 19]
     [20 21 22 23]]] # shape 3,2,4
   ```

6. os.path.join

   ```python
   >>> import os
   >>> a1 = 'home/'
   >>> a2 = 'home'
   >>> b = 'name'
   >>> print(os.path.join(a1,b))
   >>> print(os.path.join(a2,b))
   home/name
   home/name
   ```

7. os.listdir(path)

8. os.walk(); sub of sub folder recursively iterated. 

   ```python
   >>> import os
   >>> path = '/common/users/ks1418/ADE20K_2016_07_26/images/training'
   >>> for root, dirs, files in os.walk(path):
   >>>     if len(files) > 300:
   >>>         print(root)
   >>>         print(len(files))
   
   /common/users/ks1418/ADE20K_2016_07_26/images/training/s/street
   8500
   /common/users/ks1418/ADE20K_2016_07_26/images/training/p/poolroom/home
   381
   /common/users/ks1418/ADE20K_2016_07_26/images/training/o/office
   502
   
   # path: the folder to be iterated
   
   # root: the subfoler and sub of sub in 'path' that is being iterated now
   # dirs: the subfolders list in 'root'
   # the files list in 'os.path.join(root,dirs[i])'
   ```

9. os.remove( file ) : remove file

10. int division:

    ```python
    for i in range(c/10):
        pass
    # Traceback : 'float' object cannot be interpreted as an integer
    #You're creating a float as a result - to fix this use the int division operator:
    for i in range(c // 10):
        pass
    ```

    