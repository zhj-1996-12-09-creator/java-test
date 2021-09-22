package com.company.Collections;


import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/** 题目：请按照给出数据，找出同时满足以下条件的用户 * 也即以下条件：
 * * 1、全部满足偶数ID
 * * 2、年龄大于24
 * * 3、用户名转为大写
 * * 4、用户名字母倒排序
 * * 5、只输出一个用户名字 limit **/
public class StreamDemo {

    public static void main(String[] args) {
        User u1 = new User(11, "a", 23);
        User u2 = new User(12, "b", 24);
        User u3 = new User(13, "c", 22);
        User u4 = new User(14, "d", 28);
        User u5 = new User(16, "e", 26);

        List<User> list = Arrays.asList(u1, u2, u3, u4, u5);

        /** 1. 首先我们需要将 list 转化为stream流
         * 2. 然后将用户过滤出来，这里用到一个函数式接口Predicate<? super T>，我们可 以使用lambda表达式简化 *
         * 3. 这里面传递的参数，就是Stream流的泛型类型，也就是User，所以，这里可以直接 返回用户id为偶数的用户信息;
         * 4. 通过forEach进行遍历，直接简化输出 System.out::println ，等价于 System.out.println(u); **/

        List<String> collect = list.stream().filter(u -> {
            return u.getId() % 2 == 0;
        })
                .filter(u -> {
                    return u.getAge() > 24;
                })
                .map(u -> {
                    return u.getUserName().toUpperCase();
                })
                .sorted((o1, o2) -> {
                    return o2.compareTo(o1);
                })
                .limit(1)
                .collect(Collectors.toList());
        System.out.println(collect);


    }

}
