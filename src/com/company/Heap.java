package com.company;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

/**
 * @author zhanghaijie
 */
public  class Heap<T extends Comparable<T>> {
        private T[] data;
        //堆中元素的个数
        private Integer N;

    public Heap(int capacity) {
        this.data = (T[]) new Comparable[capacity+1];
        this.N = 0;
    }
    //根据堆中元素的索引获得其左孩子节点的索引
    private int leftChild(int index) {
        return index * 2 + 1;
    }

    //根据堆中元素的索引获得其右孩子节点的索引
    private int rightChild(int index) {
        return index * 2 + 2;
    }
    //交换两个元素的值
    private void exch(int i,int j){
        T temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
    private boolean less(int i,int j){
        return data[i].compareTo(data[j])<0;
    }
    //上浮算法
    private void siftUp(int k){
        //通过循环，不断的比较当前结点的值和其父结点的值，如果发现父结点的值比当前结点的值小，则交换位置
        while(k>1){
            //比较当前结点和其父结点

            if (less(k/2,k)){
                exch(k/2,k);
            }

            k = k/2;
        }
    }

    public void add(T e) {
        //总是直接插入到堆的末尾处
        data[++N]=e;
        //上浮算法进行排序
        siftUp(N);
    }

    public T delMax(){
        T max = data[1];
        //总是删除的是堆顶的数据
        if(N<= 0){
            return null;
        }

        //交换堆顶和最后一个的数据
        exch(1,N);
        data[N] = null;
        //下沉对堆顶
        N--;
        siftDown(1);
        //元素个数减1

        return max;
    }
    //下沉算法
    public void siftDown(Integer index){
        while (index*2<=N){
            int maxIndex;
            if (index*2+1>N){
                maxIndex = index*2;
            }else{
                //找到左右子节点的值最大的一个
                maxIndex = data[index*2].compareTo(data[index*2+1])<0?index*2+1:index*2;
            }
            if (data[index].compareTo(data[maxIndex])>0){
                break;
            }
            //交换最大索引处的值和当前索引
            exch(index,maxIndex);
            //
            index = maxIndex;
        }

    }

    public static void main(String[] args) {
        Heap<String> heap = new Heap<String>(9);
        heap.add("A");
        heap.add("B");
        heap.add("C");
        heap.add("D");
        heap.add("E");
        heap.add("F");
        heap.add("G");


        //通过循环从堆中删除数据
        String result = null;
        while((result = heap.delMax())!=null){
            System.out.print(result+" ");
        }
    }
}
