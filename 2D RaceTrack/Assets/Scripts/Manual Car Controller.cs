using UnityEngine;

public class ManualCarController : MonoBehaviour
{
    public float speed;
    public float turnSpeed;
    private Rigidbody2D rb;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Car car = GetComponent<Car>();
        speed = car.speed;
        turnSpeed = car.turnSpeed;
        rb = car.rb;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        float move = Input.GetAxis("Vertical") * speed * Time.deltaTime;
        float turn = Input.GetAxis("Horizontal") * turnSpeed * Time.deltaTime;

        rb.AddForce(transform.up * move, ForceMode2D.Force);
        //rb.MoveRotation(rb.rotation - turn);
        rb.AddTorque(-turn * turnSpeed * Time.deltaTime, ForceMode2D.Force);
    }
}
