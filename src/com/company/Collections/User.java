package com.company.Collections;


public class User {
    private int id;
    private String userName;
    private int age;

    public String getUserName() {
        return userName;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }



    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    //get、set、有参/无参构造器、toString
    public User (int id,String userName,int age){
        this.id = id;
        this.userName = userName;
        this.age = age;
    }

}