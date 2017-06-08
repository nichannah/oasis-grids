
program ice

use coupler, only : coupler_init, coupler_init_done, coupler_close, &
                    coupler_get, coupler_put, coupler_add_field, &
                    coupler_destroy_field, couple_field_type, &
                    COUPLER_IN, coupler_dump_field

implicit none

    type(couple_field_type), dimension(:), allocatable :: fields
    integer :: i, j, t, timestep
    logical :: debug
    real :: time_start, time_finish

    debug = .false.
    timestep = 1

    ! Initialise the coupler.
    call cpu_time(time_start)
    call coupler_init('icexxx', 1440, 1080, 1, 1)

    ! Create/add the coupling field.
    allocate(fields(1))
    call coupler_add_field(fields(1), 'dest_field', COUPLER_IN)
    call coupler_init_done()
    call cpu_time(time_finish)

    print*, "Ice startup time: ", time_finish - time_start

    ! Get fields from coupler and write out.
    call cpu_time(time_start)
    do t=1,1
      call coupler_get(timestep*t, fields)
      if (maxval(fields(1)%field) /= t) then
        stop 'Bad coupling field'
      endif
    enddo

    if (debug) then
      call coupler_dump_field(fields(1), 'dest_field.nc')
    endif

    call coupler_destroy_field(fields(1))
    call coupler_close()
    call cpu_time(time_finish)

    print*, "Ice runtime: ", time_finish - time_start

    deallocate(fields)

end program ice
