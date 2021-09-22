package com.company;

import java.util.Arrays;
import java.util.Collection;

/**
 * @author zhanghaijie
 */
public class ArrayList<E>{

    //初始化容量
    private static final Integer DEFAULT_CAPACITY = 10;

    private static final Object[] EMPTY_ELEMENTDATA = {};

    private static final Object[] DEFAULTCAPACITY_EMPTY_ELEMENTDATA = {};
    //数组的大小数组中包含元素的个数
    private int size;
    //记录数组是否被调整过结构
    private int modCount = 0;

    transient Object[] elementData; // non-private to simplify nested class access
    //
    private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

    /**
     * 初始化指定容量的数组
     * @param n
     */
    public ArrayList(int n) {
        if (n>0){
            this.elementData = new Object[n];
        }
        else if (n==0){
            this.elementData = EMPTY_ELEMENTDATA;
        }else{
            throw new IllegalArgumentException("Illegal Capacity: "+n);
        }
    }

    /**
     * 无参的构造方法
     */
    public ArrayList(){
        this.elementData = DEFAULTCAPACITY_EMPTY_ELEMENTDATA;
    }
    //指定初始化数据初始化
    public ArrayList(Collection<? extends E> c){
        //将传入的数组引用赋值给elementData
        this.elementData = c.toArray();
        if((size = elementData.length)!=0){
            if (elementData.getClass() != Object[].class){
                elementData = Arrays.copyOf(elementData,size,Object[].class);
            }
        }else{
            this.elementData = EMPTY_ELEMENTDATA;
        }

    }

    public Object[] toArray() {
        return Arrays.copyOf(elementData, size);
    }

    /**
     * 将该<tt> ArrayList </ tt>实例的容量调整为*列表的当前大小。应用程序可以使用此操作来最大程度地减少<tt> ArrayList </ tt>实例的存储。
     */
    public void trimToSize(){
        modCount++;
        if (size < elementData.length){
            elementData = (size == 0)
                    ? EMPTY_ELEMENTDATA
                    //重新进行cpoy，size的容量
                    : Arrays.copyOf(elementData, size);
        }
    }

    public int indexOf(Object o){
        if (o == null){
            for (int i=0;i<size;i++){
                if (elementData[i] == null){
                    return i;
                }
            }
        }else{
            for (int i=0;i<size;i++){
                if (o.equals(elementData[i])){
                    return i;
                }
            }
        }
        return -1;
    }

    public int lastIndexOf(Object o){
        if (o == null){
            for (int i=size ;i>0;i--){
                if (elementData[i] == null){
                    return i;
                }
            }
        }else{
            for (int i=size ;i>0;i--){
                if (o.equals(elementData[i])){
                    return i;
                }
            }
        }
        return -1;
    }

    public boolean add(E e) {
        //判断是否需要扩容
        ensureCapacityInternal(size + 1);  // Increments modCount!!
        elementData[size++] = e;
        return true;
    }

    public void ensureCapacity(int minCapacity) {
        int minExpand = (elementData != DEFAULTCAPACITY_EMPTY_ELEMENTDATA)
                // any size if not default element table
                ? 0
                // larger than default for default empty table. It's already
                // supposed to be at default size.
                : DEFAULT_CAPACITY;

        if (minCapacity > minExpand) {
            ensureExplicitCapacity(minCapacity);
        }
    }

    private void ensureCapacityInternal(int minCapacity) {
        if (elementData == DEFAULTCAPACITY_EMPTY_ELEMENTDATA) {
            minCapacity = Math.max(DEFAULT_CAPACITY, minCapacity);
        }

        ensureExplicitCapacity(minCapacity);
    }

    private void ensureExplicitCapacity(int minCapacity) {
        modCount++;

        // overflow-conscious code
        if (minCapacity - elementData.length > 0) {
            grow(minCapacity);
        }
    }
    /**
     * 扩容的方法
     * @param minCapacity
     */
    private void grow(int minCapacity) {
        // overflow-conscious code
        int oldCapacity = elementData.length;
        //公式 旧值+旧值/2
        int newCapacity = oldCapacity + (oldCapacity >> 1);
        //扩容后的值比期望值还小，要期望值
        if (newCapacity - minCapacity < 0) {
            newCapacity = minCapacity;
        }
        //扩容后的值比数组允许最大值还要大抛异常
        if (newCapacity - MAX_ARRAY_SIZE > 0) {
            newCapacity = hugeCapacity(minCapacity);
        }
        // minCapacity is usually close to size, so this is a win:
        elementData = Arrays.copyOf(elementData, newCapacity);
    }
    private static int hugeCapacity(int minCapacity) {
        if (minCapacity < 0) // overflow
        {
            throw new OutOfMemoryError();
        }
        return (minCapacity > MAX_ARRAY_SIZE) ?
                Integer.MAX_VALUE :
                MAX_ARRAY_SIZE;
    }

    public E remove(int index) {
        rangeCheck(index);

        modCount++;
        E oldValue = elementData(index);

        int numMoved = size - index - 1;
        if (numMoved > 0) {
            System.arraycopy(elementData, index+1, elementData, index,
                    numMoved);
        }
        elementData[--size] = null; // clear to let GC do its work

        return oldValue;
    }

    private void rangeCheck(int index) {
        if (index >= size) {
            throw new IndexOutOfBoundsException(outOfBoundsMsg(index));
        }
    }
    private String outOfBoundsMsg(int index) {
        return "Index: "+index+", Size: "+size;
    }


    private E elementData(int index) {
        E e = (E) elementData[index];
        return e;
    }

    public void clear (){
        modCount++;
        // clear to let GC do its work
        for (int i = 0;i<size ;i++){
            elementData[i]=null;
        }
        size =0;
    }


}
