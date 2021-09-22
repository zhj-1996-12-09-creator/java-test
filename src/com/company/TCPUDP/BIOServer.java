package com.company.TCPUDP;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class BIOServer {

    public static void main(String[] args) throws IOException {

        Socket socket = null;
        ServerSocket server = null;
        InputStream in = null;
        OutputStream out = null;
        try {
            server = new ServerSocket(8080);
            System.out.println("service端启动成功，监听端口是8080，等待客户端连接");
            while (true){
                //等待客户端连接
                socket = server.accept();
                System.out.println("客户端连接成功，连接地址"+socket.getRemoteSocketAddress());
                in = socket.getInputStream();
                //读取客户端信息
                byte[] buffer = new byte[1024];
                int len = 0;
                while ((len = in.read(buffer)) >0){
                    System.out.println(new String(buffer, 0, len));
                }
                //发送客户端信息
                out = socket.getOutputStream();
                out.write("hello!".getBytes());
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            socket.close();
            server.close();
            in.close();
            out.close();
        }
    }

}
