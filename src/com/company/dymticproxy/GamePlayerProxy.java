package com.company.dymticproxy;

public class GamePlayerProxy implements IGamePlayer{

    private IGamePlayer gamePlayer = null;
    //代练的对象
    public GamePlayerProxy(GamePlayer gamePlayer){
        this.gamePlayer = gamePlayer;
    }


    @Override
    public void login(String userName, String password) {
        this.gamePlayer.login(userName,password);
    }

    @Override
    public void killBoss() {
        System.out.println("代练打怪中");
        this.gamePlayer.killBoss();
    }

    @Override
    public void upgrade() {
        System.out.println("代练升级中");
        this.gamePlayer.upgrade();
    }
}
