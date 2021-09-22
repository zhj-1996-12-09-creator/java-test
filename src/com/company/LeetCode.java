/*
package com.company;

import org.omg.CORBA.INTERNAL;

import java.util.*;

public class LeetCode {
    */
/**
     * 求一个数组中，两个数之和等于一个标值
     *//*

    public Integer[] computeSumTwoNumber(Integer [] a, int value){
        Map<Integer,Integer> map = new HashMap<Integer,Integer>();
        String index = "";
        List<Integer> list = new ArrayList<Integer>();
        for (Integer i=0;i<a.length;i++){
            Integer b =value - a[i];
            if (index.contains(i.toString())){
                continue;
            }
            for (Integer j=0;j<a.length;j++){
                if (b.equals(a[j])){
                    */
/*if ("".equals(index)){
                        index += i.toString()+j.toString();
                    }else{
                        index += ","+i.toString()+j.toString();
                    }*//*

                    list.add(i);
                    list.add(j);
                }
            }
        }
        Integer[] arr =list.toArray(new Integer[list.size()]);
        return arr;
    }

    */
/**
     * 两个数的乘积是某个值
     * @param
     *//*

    public String computeAndTwoNumber(Integer [] a, int value){
        Map<Integer,Integer> map = new HashMap<Integer,Integer>();
        String index = "";
        for (Integer i=0;i<a.length;i++){
            if (index.contains(i.toString())){
                continue;
            }
            for (Integer j=0;j<a.length;j++){
                if (a[i]*a[j]==value){
                    if ("".equals(index)){
                        index += i.toString()+j.toString();
                    }else{
                        index += ","+i.toString()+j.toString();
                    }
                }
            }
        }
        return index;
    }
    public int[] twoSum(int [] nums , int b){
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0 ;i<nums.length;i++){
            if (map.containsKey(b-nums[i])){
                return new int[]{
                        map.get(b-nums[i]),i
                };
            }
            map.put(nums[i],i);
        }
        return null;
    }

    */
/**
     * 一个字符串中最大不重复的字串
     * @param
     *//*

    public int getMaxString(String aims) {
        int left = 0;
        int right = 1;
        int max = 0;
        Map<Object,Integer> map = new HashMap<>();
        for (;right<aims.length();right++){
            char s = aims.charAt(right);
            if (map.containsKey(s)){
                left++;
                max = Math.max(right-left-1,max);
            }
            map.put(s,right);
            max = Math.max(right-left,max);
        }
        return max;
    }

    */
/**
     * 整数反转
     * @param
     *//*

    public int resverInt(int arm){
        String armStr = Integer.toString(arm);
        int left =0;
        int right = armStr.length()-1;
        if (arm <0){
            left =1;
        }
        Map<Integer,Object> map = new HashMap<>();
        for (;right-left>=0;){
            if ('0'==armStr.charAt(left)){
                left++;
                continue;
            }
            if ('0'==armStr.charAt(right)){
                right--;
                continue;
            }
            map.put(left,armStr.charAt(right));
            map.put(right,armStr.charAt(left));
            left++;
            right--;
        }
        StringBuilder sb = new StringBuilder();
        if (arm<0){
            sb.append("-");
        }
        for (Integer key:map.keySet()){
            sb.append(map.get(key));
        }
        int result = Integer.valueOf(sb.toString());
        return result;
    }

    */
/**
     * 正则匹配
     * @param
     *//*

    public boolean pq(String str,String ptr){
        int start = 0;

        boolean result = true;
        if(ptr.indexOf("*")==0){
            start=ptr.length()-1;
            int i=str.length()-1;
            for (;start>=1;start--){
                if (str.charAt(i)!=ptr.charAt(start)){
                    result = false;
                }
                i--;
            }
        }else if (ptr.indexOf("*")==ptr.length()-1){
            for (;start<ptr.length()-1;start++){
                if (str.charAt(start)!=ptr.charAt(start)){
                    result = false;
                }
            }
        }else if(ptr.indexOf("*")==-1){
            for (;start<ptr.length();start++){
                if (str.charAt(start)!=ptr.charAt(start)){
                    result = false;
                }
            }
        }else{
            start = ptr.indexOf("*");
            for (int i=0;i<start;i++){
                if (str.charAt(i)!=ptr.charAt(i)){
                    result = false;
                }
            }
            int p = ptr.length()-1;
            for (int i=str.length()-1;i>str.length()-(ptr.length()-start);i--){
                if (str.charAt(i)!=ptr.charAt(p)){
                    result = false;
                }
                p--;
            }
        }
        return result;
    }


    public static void main(String[] args) {
        LeetCode leetCode = new LeetCode();
        int []a = {5,2,7,0,15};
        Integer[] index =leetCode.computeAndTwoNumber(a,15);
        System.out.println(index);
        for (int i =0;i<index.length;i++){
            System.out.println(index[i]);
        }
        */
/*int [] f = leetCode.twoSum(a,7);
        for (int i =0;i<f.length;i++){
            System.out.println(f[i]);
        }*//*

        */
/*String h = "abcabcabc";
        int g = leetCode.getMaxString(h);
        System.out.println(g);*//*

       // System.out.println(leetCode.resverInt(-23108));
       */
/* String p = "agjjjjga";
        String q = "agjjjjga";
        System.out.println(leetCode.pq(p,q));*//*

    }
}
*/
