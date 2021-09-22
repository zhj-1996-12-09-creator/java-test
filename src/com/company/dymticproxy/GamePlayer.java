package com.company.dymticproxy;

public class GamePlayer implements IGamePlayer{

    private String name;
    public GamePlayer(String userName){
        this.name = userName;
    }

    @Override
    public void login(String userName, String password) {
        System.out.println("登录成功"+userName);
    }

    @Override
    public void killBoss() {
        System.out.println("终极大boss");
    }

    @Override
    public void upgrade() {
        System.out.println("升级中");
    }
}
