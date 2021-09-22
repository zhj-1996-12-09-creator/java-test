package com.company.sort;

public class sort {


    static class Bolle{

        public Bolle(){

        }

        public void sort(Comparable [] a){
            System.out.println(a.length);
            for (int i=a.length-1;i>0;i--){
                System.out.println("-------------");
                for (int j=0;j<i;j++){
                    if(a[j].compareTo(a[j+1])>0){
                        System.out.println("第"+i+"次"+a[j]+a[j+1]);
                        Comparable temp = a[j];
                        a[j] = a[j+1];
                        a[j+1] =temp;
                    }
                }
            }
        }
    }
    public boolean greater(Comparable a,Comparable b){
        return a.compareTo(b)>0;
    }
    public void exchange(Comparable [] a,int i,int j){
        Comparable temp = null;
        temp = a[i];
        a[i]=a[j];
        a[j]=temp;
    }
    static class Selection{
        public void sort(Comparable [] a){
            for (int i=0;i<a.length-2;i++){
                int minIndex = i;
                for (int j =i+1;j<a.length;j++){
                    if (new sort().greater(a[i],a[j])){
                        minIndex = j;
                    }
                }
                new sort().exchange(a,i,minIndex);
            }
        }
    }

    static class Insertion{
        public void sort(Comparable [] a){
            for (int i=1;i<a.length;i++){
                for (int j=i;j>0;j--){
                    if (new sort().greater(a[j-1],a[j])){
                        new sort().exchange(a,j-1,j);
                    }
                }
            }
        }
    }


    static class shell{
        public void sort(Comparable [] a){
            //确定初始h
            int h = 1;
            while (h < a.length/2){
                h = h*2+1;

            }
            while (h>=1){
                for (int i =h;i<a.length;i++){
                    for (int j = i;j>=h;j-=h){
                        if (new sort().greater(a[j-h],a[j])){
                            new sort().exchange(a,j,j-h);
                        }else {
                            break;
                        }
                    }

                }
                h=h/2;
            }
        }
    }
    public void quickSort(Comparable []a){
        int lo = 0;
        int hi = a.length-1;
        sort(a,lo,hi);
    }

    public void sort(Comparable [] a ,int lo, int hi){
        //安全性校验
        if (hi<=lo){
            return;
        }

        //需要对数组中lo索引到hi索引处的元素进行分组（左子组和右子组）；
        int partition = partition(a, lo, hi);//返回的是分组的分界值所在的索引，分界值位置变换后的索引

        //让左子组有序
        sort(a,lo,partition-1);

        //让右子组有序
        sort(a,partition+1,hi);

    }
    private int partition(Comparable [] a ,int lo,int po){
        //安全性校验
        if (lo >= po){
            return -1;
        }
        Comparable key = a[lo];
        //定义两个指针，分别指向待瓜分数组的最大索引和最小索引
        int left = lo;
        int right = po+1;

        while (true){
            //移动右指针，知道找到大于基值暂停
            //先从右往左扫描，移动right指针，找到一个比分界值小的元素，停止
            while(greater(key,a[--right])){
                if (right==lo){
                    break;
                }
            }

            //再从左往右扫描，移动left指针，找到一个比分界值大的元素，停止
            while(greater(a[++left],key)){
                if (left==po){
                    break;
                }
            }
            //交换左右指针两个值
            if (left>=right){
                break;
            }else{
                exchange(a,left,right);
            }
        }

        //交换分界值，此时right和left相遇
        exchange(a,lo,right);
        return left;
    }

    public static void main(String[] args) {
            Integer [] a = {2,4,1,7,3,0};
       /* Bolle bolle = new Bolle();
        //bolle.sort(a);
        Selection selection = new Selection();
        //selection.sort(a);
        Insertion insertion = new Insertion();
        //insertion.sort(a);
        shell shell = new shell();*/
       // shell.sort(a);
        sort sort = new sort();
        sort.quickSort(a);

        for (int i=0;i<a.length;i++){
            System.out.println(a[i]);
        }

    }
}
