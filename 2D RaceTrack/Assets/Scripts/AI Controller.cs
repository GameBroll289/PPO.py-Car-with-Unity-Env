using UnityEngine;
using System.IO.MemoryMappedFiles;
using System;
using System.Runtime.InteropServices;

public class AICarController : MonoBehaviour
{
    public float speed;
    public float turnSpeed;
    private Rigidbody2D rb;
    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor accessor;
    private const int slotSize = 4;
    private const int actionStart = 18;
    private const int actionCount = 2;

    void Start()
    {
        Car car = GetComponent<Car>();
        speed = car.speed;
        turnSpeed = car.turnSpeed;
        rb = car.rb;

        mmf = MemoryMappedFile.OpenExisting("unity_ram");
        accessor = mmf.CreateViewAccessor();
    }

    void FixedUpdate()
    {
        float move = ReadSlot(actionStart + 0);
        float turn = ReadSlot(actionStart + 1);

        rb.AddForce(transform.up * (move * speed * Time.deltaTime), ForceMode2D.Force);
        // rb.MoveRotation(rb.rotation - (turn * turnSpeed * Time.deltaTime));
        // rb.AddForce(transform.up * move, ForceMode2D.Force);
        rb.AddTorque(-turn * turnSpeed * Time.deltaTime, ForceMode2D.Force);
    }

    float ReadSlot(int index)
    {
        float value;
        accessor.Read(index * slotSize, out value);
        return value;
    }

    void OnApplicationQuit()
    {
        accessor?.Dispose();
        mmf?.Dispose();
    }
}
