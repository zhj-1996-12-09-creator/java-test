package com.company;

public class HeapSort {

    //判断heap堆中索引i处的元素是否小于索引j处的元素
    private static  boolean less(Comparable[] heap, int i, int j) {
        return heap[i].compareTo(heap[j])<0;
    }

    //交换heap堆中i索引和j索引处的值
    private static  void exch(Comparable[] heap, int i, int j) {
        Comparable tmp = heap[i];
        heap[i] = heap[j];
        heap[j] = tmp;
    }

    private Comparable[] creatHeap(Comparable[] source){
        Comparable[] comparables = new Comparable[source.length + 1];
        //复制一个数组（无序的）
        System.arraycopy(source,0,comparables,1,source.length);
        int N = comparables.length;
        //排序，去取数组一半处的索引，然后向上遍历，每次都对当前索引进行下沉算法
        for (int i = N/2;i>0;i--){
            sink(comparables,i,N-1);
        }
        return comparables;

    }

    private void sort(Comparable[] source){
        //创建一个对、堆
        Comparable[] heap = creatHeap(source);
        //设置一个变量，记录堆中的最大索引
        int N = heap.length-1;
        ////通过循环，交换1索引处的元素和排序的元素中最大的索引处的元素
        while (N != 1){
            exch(heap,1,N);
            N--;
            sink(heap,1,N);
        }
        System.arraycopy(heap,1,source,0,source.length);
    }

    //在heap堆中，对target处的元素做下沉，范围是0~range
    private static void sink(Comparable[] heap, int target, int range) {

        while (2 * target <= range) {
            //1.找出当前结点的较大的子结点
            int max;
            if (2 * target + 1 <= range) {
                if (less(heap, 2 * target, 2 * target + 1)) {
                    max = 2 * target + 1;
                } else {
                    max = 2 * target;
                }
            } else {
                max = 2 * target;
            }

            //2.比较当前结点的值和较大子结点的值
            if (!less(heap, target, max)) {
                break;
            }

            exch(heap, target, max);

            target = max;
        }
    }

    public static void main(String[] args) {
        Comparable[] source = {"A","B","F","H","Y"};
        HeapSort HeapSort = new HeapSort();
        HeapSort.sort(source);
        for (int i=0;i<source.length;i++){
            System.out.println(source[i]);
        }
    }
}
