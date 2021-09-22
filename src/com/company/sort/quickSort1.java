package com.company.sort;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class quickSort1 {

    public static int[] quicksortint(int [] arr,int low,int high){
        int start = low;
        int end = high;
        int key = arr[low];
        while(end>start){
            //从后向前
            while(end>start && arr[end]>=key){
                end--;
            }
            //如果没有比基准值小的，则比较下一个。直到有比基准值小的，则交换位置，然后有从前向比较
            if (arr[end]<key){
                int temp = arr[end];
                arr[end] = arr[start];
                arr[start] = temp;
            }
            //从前往后比较
            while(end>start && arr[start]<=key){
                start++;
            }
            //如果没有比基准值大的数则比较下一个，知道有比基准值大的交换位置
            if (arr[start] > key){
                int temp = arr[end];
                arr[end] = arr[start];
                arr[start] = temp;
            }

        }
        //此时第一次循环结束，基准值的位置确定，左边的都小于基准值有边都大于基准值，然后递归调用
        if (start>low) quicksortint(arr,low,start-1);
        if (end<high) quicksortint(arr,end+1,high);
        return arr;
    }

    public static int [] shellSort(int[] arr){
        int dk = arr.length/3 + 1;
        while( dk != 1){
            shellInsertSort(arr,dk);
            dk = dk/3+1;
        }
        if (dk == 1) { shellInsertSort(arr,dk);}
        return arr;
    }
    public static void shellInsertSort(int []a,int dk){
        //类似于插入排序算法，但插入排序的增量时1，这里时dk，
        for (int i = dk; i < a.length; i++) {
            if (a[i]<a[i-dk]){
                //交换位置
                int j;
                int x = a[i];
                a[i] = a[i-dk];
                for (j=i-dk;j>=0 && x<a[j];j=j-dk){
                    //网前遍历，看是不是还小
                    a[j+dk] = a[j];
                }
                a[j+dk] = x;
                //将数据插入对应的位置
            }
        }
    }

    Lock lock = new ReentrantLock();
    Condition condition1 = lock.newCondition();
    Condition condition2 = lock.newCondition();
    static int num = 0;
    public  void small(){
        lock.lock();

        try {
            char a = 'a';
            for (int i = 0; i < 26; i++) {
                while(num %2 != 0){
                    condition1.await();
                }
                System.out.print(a);   // ABCDEFGHIJKLMNOPQRSTUVWXYZ
                a++;
                num++;
                condition2.signalAll();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            lock.unlock();
        }
    }

    public void big(){
        lock.lock();
        try {
            char a = 'A';
            for (int i = 0; i < 26; i++) {
                while (num % 2 == 0){
                    condition2.await();
                }
                System.out.print(a);   // ABCDEFGHIJKLMNOPQRSTUVWXYZ
                a++;
                num++;
                condition1.signalAll();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            lock.unlock();
        }
    }

    public int fib(int n){
        if (n == 0) {
            return 0;
        }
        if (n==1){
            return 1;
        }
        return fib(n-1)*n;
    }
    public long zhishu(){
        int i,j;
        Long sum =0L;
        for (i=2;i<100;i++){
            for (j=2;j<i;j++){
                if (i%j == 0){
                    break;
                }
            }
            if(j>=i) {
                sum += fib(i);
            }
        }
        return sum;
    }

    public static void main(String[] args) {
        quickSort1 quickSort1 = new quickSort1();
        System.out.println(quickSort1.zhishu());
        /*new Thread(()->{
            quickSort1.small();
        }).start();
        new Thread(()->{
            quickSort1.big();
        }).start();
*/

    }
}
